"""
RAG nodes: recuperação, filtragem por relevância e montagem de contexto sobre
os protocolos institucionais.

Segue o padrão ensinado na Aula 03 (RAG com LangGraph): busca por MMR (Maximum
Marginal Relevance) para evitar documentos redundantes, um nó de filtragem que
usa o próprio LLM para pontuar relevância (0-10) e descarta o que não serve, e
um confidence_score derivado dessa pontuação - usado depois pelo roteamento
condicional do grafo principal para decidir se a conduta sugerida precisa de
alerta extra.

Diferença deliberada em relação ao material de referência: aqui os documentos
já nascem com metadado de fonte (protocolo/FAQ/laudo), então a filtragem por
relevância soma-se à fonte já rastreada - não precisamos reconstruir a fonte
a partir do texto.
"""

import logging
from typing import List
import re

# Monitoramento operacional do proprio pipeline de RAG (distinto dos alertas
# clinicos de guardrails.py, que vao para a equipe medica sobre o PACIENTE).
# Este logger sinaliza problemas de SAUDE DO PIPELINE em si - zero documentos
# recuperados ou confidence_score muito baixo - para quem opera o sistema,
# nao para quem le a resposta clinica. Padrao inspirado no "RAGMonitor" da
# Aula 03 de LangGraph (RAG).
_monitor_logger = logging.getLogger("rag_monitor")
if not _monitor_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s | RAG-MONITOR | %(levelname)s | %(message)s"))
    _monitor_logger.addHandler(_handler)
_monitor_logger.setLevel(logging.INFO)

CONFIDENCE_ALERTA_MONITOR = 0.15

from langchain_core.prompts import ChatPromptTemplate

from states import RAGState


def extract_text(response) -> str:
    """Extrai o texto de uma resposta de LLM, seja qual for o backend.

    Modelos de chat (ChatOllama, ChatOpenAI) retornam um objeto com `.content`.
    LLMs "puros" como HuggingFacePipeline retornam a string diretamente. Sem
    isso, o mesmo codigo quebra ao trocar de backend (--backend huggingface
    vs --backend ollama).
    """
    return response.content if hasattr(response, "content") else str(response)


def _parse_relevancia(texto: str) -> int | None:
    """Extrai a nota de relevancia (1-10) da resposta do LLM, de forma tolerante.

    Modelos pequenos rodando sem chat template (ex: HuggingFacePipeline puro,
    como o Qwen2.5 via --backend huggingface) raramente respondem no formato
    rigido "SIM:<nota>" pedido no prompt - eles tendem a explicar, parafrasear
    ou colocar a resposta no meio de um paragrafo. Por isso a busca e feita
    com regex em qualquer parte do texto, nao so no inicio, e "SIM" sem nota
    numerica ainda conta (nota padrao 5) em vez de ser descartado.
    """
    texto_upper = texto.upper()

    if re.search(r"\bNAO\b", texto_upper) and "SIM" not in texto_upper:
        return None

    match = re.search(r"SIM\s*:?\s*(\d{1,2})", texto_upper)
    if match:
        nota = int(match.group(1))
        return min(nota, 10)

    if "SIM" in texto_upper:
        # A nota pode vir separada do "SIM" (ex: "Sim, e relevante. Nota: 7")
        match_nota = re.search(r"NOTA\s*:?\s*(\d{1,2})", texto_upper)
        if match_nota:
            return min(int(match_nota.group(1)), 10)
        return 5  # relevante, mas sem nota numerica clara

    return None


RELEVANCE_PROMPT = ChatPromptTemplate.from_template(
    "Analise se o trecho abaixo e relevante para responder a pergunta do medico.\n\n"
    "Pergunta: {pergunta}\n\n"
    "Trecho: {conteudo}\n\n"
    "Responda apenas \"SIM:<nota de 1 a 10>\" se relevante, ou \"NAO\" se irrelevante."
)


def retrieve_documents(state: RAGState, knowledge_base, k: int = 10) -> RAGState:
    """Nó de recuperacao: busca por similaridade de embedding na base FAISS.

    Retorna tambem a distancia bruta de cada documento (usada depois em
    filter_and_rank como criterio PRIMARIO de relevancia). Trocamos MMR por
    similaridade pura de proposito: com um corpus pequeno (10-20 documentos),
    a diversificacao do MMR competia por vaga com documentos genuinamente mais
    relevantes, e o ganho de "evitar redundancia" nao compensava o risco de
    trazer conteudo fora do assunto.

    k subiu de 6 para 10 em 2026-09-13, quando o corpus cresceu para 71
    documentos (MedQuAD + PCDT/ILAS oficiais, ver dataset_completo.jsonl):
    com k=6, documentos oficiais que ficam um pouco mais distantes em termos
    lexicais (Secao 5, BOOST_OFICIAL_PCT) nem chegavam a ser considerados por
    filter_and_rank, que so opera sobre os k documentos ja recuperados aqui -
    nenhum boost/corte posterior resgata um documento que a busca inicial
    nem trouxe. O corte relativo em filter_and_rank continua responsavel por
    descartar o excesso de ruido que um k maior traz.
    """
    pergunta = state["pergunta"]

    resultados = knowledge_base.similarity_search_with_score(pergunta, k=k)

    raw_documents = [
        {
            "conteudo": doc.page_content,
            "fonte": doc.metadata.get("fonte", "desconhecida"),
            "fonte_url": doc.metadata.get("fonte_url", ""),
            "tipo": doc.metadata.get("tipo", "desconhecido"),
            "oficial": bool(doc.metadata.get("oficial", False)),
            "doenca": doc.metadata.get("doenca", ""),
            # FAISS retorna distancia L2 por padrao: MENOR = mais similar/relevante.
            "distancia": float(distancia),
        }
        for doc, distancia in resultados
    ]
    return {**state, "raw_documents": raw_documents}


BOOST_PROTOCOLO_PCT = 0.10  # reduz a distancia efetiva de protocolos em 10% (desempate)
BOOST_OFICIAL_PCT = 0.30  # fontes oficiais (PCDT/ILAS) - ver rationale na docstring
DISTANCIA_RELATIVA_MAXIMA = 1.45  # descarta candidatos > 45% mais distantes que o melhor colocado


def filter_and_rank(state: RAGState, llm=None, top_k: int = 4) -> RAGState:
    """Nó de filtragem/ranking: usa a distancia de embedding como criterio
    PRIMARIO de relevancia, em vez de pedir para o LLM julgar relevancia
    textualmente (formato "SIM:<nota>").

    Por que a mudanca: testes reais com modelos pequenos rodando localmente
    (ex: Qwen2.5-1.5B quantizado via Ollama) mostraram que o julgamento de
    relevancia do LLM e pouco confiavel - em um caso observado, uma FAQ sobre
    dosagem de metformina (diabetes) foi julgada "relevante" para uma pergunta
    sobre suspeita de sepse, e um documento de hipertensao venceu o proprio
    protocolo de sepse no ranking. A distancia de embedding, por ser uma
    medida matematica de similaridade semantica (nao uma interpretacao
    textual de um LLM pequeno), e um sinal bem mais estavel para este corpus
    pequeno e especializado.

    O parametro `llm` e mantido na assinatura por compatibilidade com quem
    ja chama esta funcao, mas nao e mais usado para decidir o que entra no
    contexto - so a fonte de verdade da conduta clinica (protocolo/FAQ) e a
    proximidade semantica importam aqui. A geracao da resposta final continua
    usando o LLM normalmente, no nó seguinte do fluxo.

    BOOST_PROTOCOLO_PCT: em caso de distancias parecidas, documentos tipo
    "protocolo" (fonte institucional primaria) tem uma pequena vantagem sobre
    FAQs/laudos (paráfrase derivada) - protocolos costumam ter os detalhes
    operacionais completos (ex: dose de reposicao volemica, uso de
    vasopressor) que uma FAQ pode nao mencionar.

    BOOST_OFICIAL_PCT: documentos marcados `oficial=True` (PCDT do Ministerio
    da Saude, protocolo do ILAS - ver data/raw/dataset_oficial.jsonl) recebem
    um boost maior (30%) que o de protocolo comum. Motivo medido empiricamente
    em 2026-09-13: esses textos sao redigidos em linguagem tecnica/formal
    (ex: "coleta de lactato arterial") que nao repete os termos EXATOS da
    query enriquecida (ex: "Lactato serico") tao bem quanto os documentos
    sinteticos originais - por isso ficam em distancias brutas mais altas
    (6a-14a posicao) mesmo sendo a fonte mais autoritativa. Um boost de 10%
    (igual ao de protocolo comum) nao era suficiente para sequer cruzar o
    corte relativo em alguns casos (ex: PCDT de hipertensao precisava de
    ~25% so para entrar como candidato).

    Diferenca deliberada em relacao ao boost de protocolo comum: aqui o boost
    TAMBEM se aplica ao teste de corte relativo (`_distancia_para_corte`
    abaixo), nao so a ordem - ao contrario da regra geral.

    ATENCAO - bug real ja encontrado e corrigido nesta mesma funcao: a
    primeira versao aplicava o boost oficial para QUALQUER documento
    `oficial=True`, sem checar se a doenca do documento batia com a da
    query. Testado empiricamente em 2026-09-13: para a query do paciente de
    sepse, o PCDT de HIPERTENSAO (documento errado) tinha distancia bruta
    baixa o suficiente (vocabulario clinico/burocratico generico compartilhado
    entre PCDTs de doencas diferentes) para, com o boost, cruzar o corte e
    expulsar o proprio protocolo de sepse do contexto - reintroduzindo
    exatamente o bug de context bleed que motivou toda a Secao 7 do relatorio
    tecnico. Ou seja: "e raro ter documento oficial de assunto errado" NAO e
    garantia suficiente por si so, porque PCDTs de doencas diferentes se
    parecem entre si na superficie lexical mesmo sem terem relacao clinica.
    Corrigido exigindo que `doc["doenca"]` (tag manual no dataset, ver
    data/raw/dataset_oficial.jsonl) apareca na propria query (que ja vem
    enriquecida com o diagnostico do paciente, ver nodes.py) antes de
    conceder o boost - ou seja, o boost so vale quando ha evidencia de que o
    documento e realmente sobre a doenca da consulta, nao so por ele ser
    "oficial" no abstrato.
    """
    raw_documents = state.get("raw_documents", [])

    if not raw_documents:
        _monitor_logger.warning(
            f"Zero documentos recuperados para a query '{state.get('pergunta', '')[:80]}...' "
            f"- indice FAISS pode estar vazio/errado ou a query nao tem nenhum match."
        )
        return {**state, "filtered_documents": [], "confidence_score": 0.0}

    pergunta_lower = state.get("pergunta", "").lower()

    def _oficial_relevante(doc) -> bool:
        # So concede o boost oficial se a doenca do documento realmente
        # aparecer na query (que ja vem enriquecida com o diagnostico do
        # paciente) - ver ATENCAO na docstring acima sobre o bug ja
        # encontrado ao confiar so na flag `oficial`.
        doenca = doc.get("doenca", "")
        return bool(doc.get("oficial")) and bool(doenca) and doenca.lower() in pergunta_lower

    def distancia_ajustada(doc):
        if _oficial_relevante(doc):
            return doc["distancia"] * (1 - BOOST_OFICIAL_PCT)
        if doc.get("tipo") == "protocolo":
            return doc["distancia"] * (1 - BOOST_PROTOCOLO_PCT)
        return doc["distancia"]

    ordenados = sorted(raw_documents, key=distancia_ajustada)

    # Corte relativo: descarta candidatos muito mais distantes que o melhor
    # colocado, em vez de sempre preencher top_k vagas. Sem isso, quando so
    # 1 documento e de fato proximo semanticamente, os outros dois "menos
    # piores" ainda entravam no contexto so por sobrar vaga - poluindo a
    # resposta do LLM com conteudo de outra condicao clinica.
    #
    # IMPORTANTE: a referencia do corte usa a distancia BRUTA (sem boost) para
    # a maioria dos documentos. Bug ja corrigido: usar a distancia ajustada
    # como referencia deixava a janela de corte artificialmente apertada
    # sempre que um documento tipo "protocolo" IRRELEVANTE (ex: checklist de
    # alta hospitalar) aparecia bem colocado - o boost dele empurrava a
    # "melhor distancia" pra baixo, e o protocolo correto (que tambem tem
    # boost, mas partia de uma distancia bruta maior) acabava excluido por
    # uma margem minima. O boost de protocolo comum so deve decidir ORDEM
    # entre quem ja passou no corte, nunca o TAMANHO do corte.
    #
    # Excecao deliberada: documentos oficiais CUJA DOENCA BATE COM A QUERY
    # (`_oficial_relevante`) usam a distancia JA com boost tambem no proprio
    # teste de corte (ver BOOST_OFICIAL_PCT acima) - sem isso, o PCDT de
    # hipertensao, por exemplo, nem seria considerado candidato para a
    # query de um paciente hipertenso, pois sua distancia bruta fica acima
    # do corte calculado sobre o melhor colocado. A checagem de doenca
    # (`_oficial_relevante`) e o que impede essa excecao de reabrir a mesma
    # falha para um documento oficial de OUTRA doenca (ver ATENCAO acima).
    # "melhor_distancia_bruta" (usada para calcular o limite do corte)
    # continua sempre bruta, para todos os documentos.
    melhor_distancia_bruta = min(doc["distancia"] for doc in raw_documents)
    limite = melhor_distancia_bruta * DISTANCIA_RELATIVA_MAXIMA

    def distancia_para_corte(doc):
        return distancia_ajustada(doc) if _oficial_relevante(doc) else doc["distancia"]

    candidatos = [doc for doc in ordenados if distancia_para_corte(doc) <= limite]
    filtered = candidatos[:top_k]

    # Heuristico, nao uma probabilidade calibrada: quanto menor a distancia
    # media dos escolhidos, maior a confianca. Serve para o roteamento
    # condicional identificar respostas baseadas em contexto fraco/distante,
    # nao para comparar entre execucoes diferentes de forma absoluta.
    distancia_media = sum(d["distancia"] for d in filtered) / len(filtered)
    confidence = max(0.0, min(1.0, 1.0 - distancia_media))

    if confidence < CONFIDENCE_ALERTA_MONITOR:
        _monitor_logger.warning(
            f"confidence_score baixo ({confidence:.2f}) para a query "
            f"'{state.get('pergunta', '')[:80]}...' - revisar se o indice cobre este caso."
        )

    return {**state, "filtered_documents": filtered, "confidence_score": confidence}


def generate_context(state: RAGState) -> RAGState:
    """Nó de montagem de contexto: combina os documentos filtrados e suas fontes.

    Quando o documento tem uma `fonte_url` real (documentos oficiais - PCDT/
    ILAS, ver data/raw/dataset_oficial.jsonl), ela e anexada ao nome da fonte
    para permitir que o medico abra o PDF original, nao so o titulo. Docs
    sinteticos/internos nao tem URL externa real, entao ficam so com o nome,
    como antes - nao inventamos um link para o que nao tem um.
    """
    filtered_documents: List[dict] = state.get("filtered_documents", [])

    if not filtered_documents:
        return {**state, "contexto": "Nenhum protocolo relevante encontrado.", "fontes": []}

    contexto = "\n\n".join(doc["conteudo"] for doc in filtered_documents)
    fontes = []
    for doc in filtered_documents:
        fonte = doc["fonte"]
        url = doc.get("fonte_url")
        if url:
            fonte = f"{fonte} ({url})"
        fontes.append(fonte)
    return {**state, "contexto": contexto, "fontes": fontes}