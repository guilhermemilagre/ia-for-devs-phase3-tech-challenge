"""
State layer: esquemas de estado compartilhados entre os nós do LangGraph.

Separado do resto por design - é o "contrato" entre os nós, como recomendado
no material da Aula 02 (LangGraph): "State layer: define esquemas tipados
rigorosos para comunicação entre nós, garantindo contratos claros".
"""

from typing import TypedDict, List, Optional


class RAGState(TypedDict, total=False):
    """Estado do sub-pipeline de RAG sobre os protocolos institucionais.

    Modelado no padrão da Aula 03 (RAG com LangGraph): retrieve -> filter ->
    context -> generate, com confidence_score e rastreabilidade das fontes.
    """
    pergunta: str
    contexto_paciente: str
    raw_documents: List[dict]          # [{"conteudo": ..., "fonte": ...}, ...]
    filtered_documents: List[dict]
    contexto: str
    fontes: List[str]
    confidence_score: float
    resposta: str
    guardrail_acionado: bool


class AssistantState(TypedDict, total=False):
    """Estado compartilhado do fluxo multiagente do assistente médico.

    Modelado no padrão da Aula 04 (Multi-Agent): campo `etapa_atual` funciona
    como máquina de estados que a função de roteamento condicional consulta
    para decidir o próximo nó; `historico_decisoes` registra, estilo ReAct
    (Thought/Action/Observation), cada decisão tomada pelo grafo - é a base
    da auditoria e da explainability.
    """
    paciente_id: int
    pergunta: str

    exames_pendentes: List[str]
    exames_criticos_pendentes: List[str]

    rag: RAGState

    conduta_sugerida: str
    fontes: List[str]
    guardrail_acionado: bool
    confianca_baixa: bool
    urgencia: str

    alertas: List[str]

    etapa_atual: str          # "inicio" | "exames_verificados" | "conduta_sugerida" | "concluido" | "erro"
    historico_decisoes: List[str]
    erro: Optional[str]


def criar_estado_inicial(paciente_id: int, pergunta: str) -> AssistantState:
    """Cria o estado inicial para uma execução do grafo do assistente."""
    return AssistantState(
        paciente_id=paciente_id,
        pergunta=pergunta,
        exames_pendentes=[],
        exames_criticos_pendentes=[],
        rag=RAGState(),
        conduta_sugerida="",
        fontes=[],
        guardrail_acionado=False,
        confianca_baixa=False,
        urgencia="",
        alertas=[],
        etapa_atual="inicio",
        historico_decisoes=[],
        erro=None,
    )
