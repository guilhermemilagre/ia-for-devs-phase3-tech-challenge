"""
Logic layer: os nós do agente principal, cada um com responsabilidade unica.

Cada nó segue o padrao ReAct (Thought -> Action -> Observation) visto na
Aula 04 (Multi-Agent): antes de agir, registra o raciocinio; depois de agir,
registra a observacao. Isso alimenta `historico_decisoes`, que funciona como
trilha de auditoria/explainability - o mesmo papel que a Aula 04 descreve para
o "relatorio executivo" de um sistema multiagente, mas aqui aplicado a decisao
clinica.

EXAMES_CRITICOS e os guardrails de seguranca sao mantidos do design original
do projeto (ninguem prescreve nada sem validacao humana).
"""

import sqlite3

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

from states import AssistantState, RAGState
from rag_nodes import retrieve_documents, filter_and_rank, generate_context, extract_text
from guardrails import check_guardrails, log_interacao

EXAMES_CRITICOS = {"Lactato serico", "Hemocultura", "Potassio"}

# Abaixo deste confidence_score, tratamos como "praticamente nenhum documento
# relevante" (zero documentos = 0.0, ou distancia media >= 0.85). Calibrado
# com os confidence_score REAIS observados em execucoes validadas como
# clinicamente corretas: paciente 1 (hipertensao) = 0.55, paciente 3 (sepse)
# = 0.22, paciente 2 (diabetes, o mais baixo dos tres corretos) = 0.16. Um
# limiar mais alto (ex: 0.3, como sugerido no material de RAG do LangGraph)
# geraria falso alarme nos proprios casos ja validados como corretos - reflexo
# direto da limitacao do embedding MiniLM documentada no relatorio tecnico
# (Secao 5.3): a distancia bruta deste corpus e naturalmente alta mesmo
# quando o documento certo e recuperado.
CONFIANCA_MINIMA = 0.15

CONDUTA_PROMPT = PromptTemplate.from_template(
    "Voce e um assistente medico de apoio a decisao clinica. "
    "Use APENAS as informacoes de contexto abaixo. "
    "Nunca prescreva uma conduta como definitiva - sempre trate como "
    "sugestao sujeita a validacao humana.\n\n"
    "IMPORTANTE: a secao de protocolos abaixo pode conter MAIS DE UM "
    "protocolo, de condicoes clinicas diferentes (isso acontece porque a "
    "busca traz os documentos mais proximos textualmente, nem sempre so o "
    "certo). Antes de responder, identifique qual(is) protocolo(s) "
    "correspondem EXATAMENTE ao diagnostico do paciente informado no "
    "Contexto do paciente abaixo. Baseie sua resposta SOMENTE nesse(s) "
    "protocolo(s) correspondente(s) - ignore completamente qualquer "
    "protocolo de outra condicao clinica, mesmo que ele apareca no "
    "contexto. Nao mencione nem cite exames ou condutas de um protocolo "
    "que nao seja o do diagnostico do paciente.\n\n"
    "### Contexto do paciente:\n{contexto_paciente}\n\n"
    "### Protocolos institucionais relevantes:\n{contexto_protocolos}\n\n"
    "### Pergunta do medico:\n{pergunta}\n\n"
    "### Resposta:"
)


class ClassificacaoUrgencia(BaseModel):
    """Schema de saida estruturada para a classificacao de urgencia (PydanticOutputParser)."""

    nivel: str = Field(description="Nivel de urgencia: 'baixa', 'media' ou 'alta'")
    justificativa: str = Field(description="Uma frase curta justificando o nivel escolhido")


_urgencia_parser = PydanticOutputParser(pydantic_object=ClassificacaoUrgencia)

# Nao usamos `_urgencia_parser.get_format_instructions()` (o padrao do
# PydanticOutputParser) de proposito: ele despeja o JSON SCHEMA da classe
# inteiro no prompt, e testado empiricamente contra o modelo local (Qwen2.5
# fine-tuned via Ollama) o resultado foi o LLM aninhar os campos dentro de
# uma chave "properties" (copiando a estrutura do schema, nao uma instancia
# dele) - mesma categoria de problema ja visto no bug de context bleed do RAG
# (modelo pequeno nao segue formato rigido/verboso de forma confiavel). Uma
# instrucao curta com o formato final direto, sem expor o schema, funcionou
# de forma consistente nos testes (3/3 respostas validas).
URGENCIA_PROMPT = PromptTemplate(
    template=(
        "Classifique a urgencia do caso clinico abaixo em baixa, media ou alta, "
        "considerando o diagnostico, os exames criticos pendentes e a conduta sugerida.\n\n"
        "### Contexto do paciente:\n{contexto_paciente}\n\n"
        "### Exames criticos pendentes:\n{exames_criticos}\n\n"
        "### Conduta sugerida:\n{conduta}\n\n"
        "Responda APENAS com um objeto JSON, sem nenhum texto antes ou depois, "
        'exatamente neste formato: {{"nivel": "baixa", "justificativa": "..."}} '
        "- onde nivel e uma das strings baixa, media ou alta."
    ),
    input_variables=["contexto_paciente", "exames_criticos", "conduta"],
)


def _classificar_urgencia(llm, contexto_paciente: str, exames_criticos: list[str], conduta: str) -> str:
    """Classifica a urgencia do caso em baixa/media/alta usando saida estruturada
    (PydanticOutputParser, Aula 01 de LangChain na Pratica).

    Modelos pequenos rodando localmente (ja visto no bug de context bleed do
    RAG) nao seguem formatos rigidos de forma confiavel - por isso o parsing
    e protegido por try/except com fallback explicito, em vez de deixar a
    excecao subir e quebrar a sugestao de conduta (que ja foi gerada e e a
    parte crítica do fluxo). Uma falha aqui nunca deve impedir a resposta
    principal de chegar ao medico.
    """
    try:
        chain = URGENCIA_PROMPT | llm | StrOutputParser()
        resposta = chain.invoke(
            {
                "contexto_paciente": contexto_paciente,
                "exames_criticos": ", ".join(exames_criticos) or "nenhum",
                "conduta": conduta,
            }
        )
        classificacao = _urgencia_parser.parse(resposta)
        nivel = classificacao.nivel.strip().lower()
        return nivel if nivel in {"baixa", "media", "alta"} else "nao classificado"
    except Exception:
        return "nao classificado"


def _get_patient_context(db_path: str, paciente_id: int) -> tuple[str, str | None]:
    """Retorna (texto formatado do contexto do paciente, diagnostico_principal).

    O diagnostico e retornado separadamente porque tambem e usado para
    enriquecer a query de busca do RAG (ver make_sugerir_conduta_node) - sem
    isso, a pergunta generica do medico ("qual a conduta?") nao da nenhuma
    pista semantica de qual protocolo e relevante.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute(
        "SELECT nome_iniciais, idade, sexo, diagnostico_principal "
        "FROM pacientes WHERE id = ?",
        (paciente_id,),
    )
    paciente = cur.fetchone()
    if not paciente:
        conn.close()
        return "Paciente nao encontrado na base.", None

    nome, idade, sexo, diagnostico = paciente

    cur.execute(
        "SELECT nome_medicacao, dose FROM medicacoes_em_uso WHERE paciente_id = ?",
        (paciente_id,),
    )
    medicacoes = cur.fetchall()
    conn.close()

    meds_txt = "\n".join(f"- {med}: {dose}" for med, dose in medicacoes) or "Nenhuma."
    contexto = (
        f"Paciente {nome}, {idade} anos, {sexo}. Diagnostico: {diagnostico}.\n"
        f"Medicacoes em uso:\n{meds_txt}"
    )
    return contexto, diagnostico


# ---------------------------------------------------------------------------
# Nó 1 - verificar exames pendentes
# ---------------------------------------------------------------------------

def make_verificar_exames_node(db_path: str):
    def verificar_exames_pendentes(state: AssistantState) -> AssistantState:
        historico = state.get("historico_decisoes", [])
        historico.append(
            f"Thought: preciso checar se ha exames pendentes para o paciente "
            f"{state['paciente_id']} antes de sugerir qualquer conduta."
        )

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT nome_exame FROM exames WHERE paciente_id = ? AND status = 'pendente'",
            (state["paciente_id"],),
        )
        pendentes = [row[0] for row in cur.fetchall()]
        conn.close()

        criticos = [e for e in pendentes if e in EXAMES_CRITICOS]

        historico.append(
            f"Action: consultei a base de prontuarios. "
            f"Observation: {len(pendentes)} exame(s) pendente(s)"
            + (f", sendo critico(s): {', '.join(criticos)}." if criticos else ".")
        )

        return {
            **state,
            "exames_pendentes": pendentes,
            "exames_criticos_pendentes": criticos,
            "etapa_atual": "exames_verificados",
            "historico_decisoes": historico,
        }

    return verificar_exames_pendentes


# ---------------------------------------------------------------------------
# Nó 2 - sugerir conduta (RAG + LLM + guardrails)
# ---------------------------------------------------------------------------

def construir_query_enriquecida(pergunta: str, diagnostico: str | None, exames_pendentes: list[str]) -> str:
    """Enriquece a query de busca do RAG com o diagnostico do paciente e os
    nomes dos exames pendentes.

    Extraida para funcao propria para que
    `assistant/test_fix_context_bleed.py` possa testar a MESMA logica usada
    pelo pipeline real, em vez de uma copia que pode ficar desatualizada.

    A pergunta do medico costuma ser generica ("qual a conduta?"), sem
    mencionar o diagnostico - isso faz a busca semantica trazer protocolos
    irrelevantes (ex: puxar o checklist de alta hospitalar em vez do
    protocolo de sepse para um paciente septico). Enriquecer a query com o
    diagnostico do paciente ajuda, mas nao bastou sozinho em testes reais: o
    checklist de alta hospitalar tambem fala de "exames pendentes" em termos
    genericos e competia de perto com o protocolo certo. Os NOMES dos exames
    pendentes (ex: "Lactato serico", "Hemocultura") sao termos altamente
    especificos que aparecem quase literalmente no texto do protocolo de
    sepse - inclui-los na query da um sinal de similaridade textual muito
    mais forte e discriminativo do que so o nome do diagnostico.
    """
    partes_extra = []
    if diagnostico:
        partes_extra.append(f"Diagnostico do paciente: {diagnostico}.")
    if exames_pendentes:
        partes_extra.append(f"Exames pendentes: {', '.join(exames_pendentes)}.")
    if not partes_extra:
        return pergunta
    return f"{pergunta} {' '.join(partes_extra)}"


def make_sugerir_conduta_node(knowledge_base, llm, db_path: str):
    def sugerir_conduta(state: AssistantState) -> AssistantState:
        historico = state.get("historico_decisoes", [])
        historico.append(
            "Thought: nao ha bloqueio por exame critico impeditivo, entao posso "
            "buscar protocolos institucionais relevantes e propor uma sugestao de conduta."
        )

        try:
            contexto_paciente, diagnostico = _get_patient_context(db_path, state["paciente_id"])

            query_busca = construir_query_enriquecida(
                state["pergunta"], diagnostico, state.get("exames_pendentes", [])
            )

            rag_state: RAGState = {"pergunta": query_busca}
            rag_state = retrieve_documents(rag_state, knowledge_base)
            rag_state = filter_and_rank(rag_state, llm)
            rag_state = generate_context(rag_state)

            # Chain LCEL (prompt | llm | parser) em vez de prompt.format() +
            # llm.invoke() manual - mesmo resultado, mas demonstra o padrao de
            # composicao de Chains ensinado na Aula 04 de LangChain na Pratica.
            conduta_chain = CONDUTA_PROMPT | llm | StrOutputParser()
            resposta_bruta = extract_text(
                conduta_chain.invoke(
                    {
                        "contexto_paciente": contexto_paciente,
                        "contexto_protocolos": rag_state["contexto"],
                        "pergunta": state["pergunta"],
                    }
                )
            )
            resposta_final, guardrail_acionado = check_guardrails(resposta_bruta)

            confidence_score = rag_state.get("confidence_score", 0.0)
            confianca_baixa = confidence_score < CONFIANCA_MINIMA

            urgencia = _classificar_urgencia(
                llm,
                contexto_paciente,
                state.get("exames_criticos_pendentes", []),
                resposta_final,
            )

            log_interacao(
                pergunta=state["pergunta"],
                resposta=resposta_final,
                fontes=rag_state.get("fontes", []),
                paciente_id=state["paciente_id"],
                guardrail_acionado=guardrail_acionado,
                confidence_score=confidence_score,
                urgencia=urgencia,
            )

            historico.append(
                f"Action: recuperei e filtrei protocolos (confidence_score="
                f"{confidence_score:.2f}), gerei a sugestao com o LLM e classifiquei "
                f"urgencia='{urgencia}'. "
                f"Observation: {'guardrail de seguranca foi acionado' if guardrail_acionado else 'resposta dentro dos limites de atuacao'}"
                f"{'; confianca de recuperacao baixa' if confianca_baixa else ''}."
            )

            return {
                **state,
                "rag": rag_state,
                "conduta_sugerida": resposta_final,
                "fontes": rag_state.get("fontes", []),
                "guardrail_acionado": guardrail_acionado,
                "confianca_baixa": confianca_baixa,
                "urgencia": urgencia,
                "etapa_atual": "conduta_sugerida",
                "historico_decisoes": historico,
            }
        except Exception as e:
            historico.append(f"Observation: erro ao sugerir conduta: {e}")
            return {
                **state,
                "erro": str(e),
                "etapa_atual": "erro",
                "historico_decisoes": historico,
            }

    return sugerir_conduta


# ---------------------------------------------------------------------------
# Nó 3 - emitir alertas
# ---------------------------------------------------------------------------

def emitir_alertas(state: AssistantState) -> AssistantState:
    historico = state.get("historico_decisoes", [])
    historico.append("Thought: preciso consolidar os alertas para a equipe medica.")

    alertas = []
    criticos = state.get("exames_criticos_pendentes", [])
    if criticos:
        alertas.append(
            f"ALERTA: exame(s) critico(s) pendente(s) para o paciente "
            f"{state['paciente_id']}: {', '.join(criticos)}."
        )
    if state.get("guardrail_acionado"):
        alertas.append(
            f"ALERTA: sugestao para o paciente {state['paciente_id']} acionou "
            f"guardrail de seguranca e requer revisao humana prioritaria."
        )
    if state.get("confianca_baixa"):
        confidence_score = state.get("rag", {}).get("confidence_score", 0.0)
        alertas.append(
            f"ALERTA: confianca de recuperacao do RAG baixa (confidence_score="
            f"{confidence_score:.2f}) para o paciente {state['paciente_id']} - "
            f"revisar as fontes com atencao redobrada antes de validar esta sugestao."
        )
    if state.get("urgencia") == "alta":
        alertas.append(
            f"ALERTA: caso do paciente {state['paciente_id']} classificado como "
            f"urgencia ALTA - priorizar revisao médica."
        )

    historico.append(f"Action: alerta(s) consolidado(s). Observation: {len(alertas)} alerta(s) gerado(s).")

    return {
        **state,
        "alertas": alertas,
        "etapa_atual": "concluido",
        "historico_decisoes": historico,
    }