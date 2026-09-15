"""
Orchestration layer: constroi o StateGraph com roteamento condicional real.

Padrao da Aula 04 (Multi-Agent): uma funcao `deve_continuar` consulta
`etapa_atual` no estado e decide qual no roda a seguir - em vez de um
`add_edge` fixo entre cada par de nos. Isso e o que diferencia LangGraph de
um pipeline linear: o proximo passo depende do estado, nao da ordem em que
os nos foram declarados.

Tambem usa `MemorySaver` como checkpointer (persistencia de estado), como
ensinado na mesma aula - permite retomar/inspecionar uma execucao pelo
`thread_id`, util para auditoria e para recuperacao apos falha.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from states import AssistantState
from nodes import make_verificar_exames_node, make_sugerir_conduta_node, emitir_alertas


def _deve_continuar(state: AssistantState) -> str:
    """Roteamento condicional baseado no estado atual do fluxo.

    Importante: mesmo quando um no anterior sinaliza erro, o fluxo passa por
    `emitir_alertas` antes de encerrar - um alerta de exame critico pendente
    nao pode depender de a sugestao de conduta ter dado certo. `emitir_alertas`
    sempre termina marcando `etapa_atual = "concluido"`, o que evita loop.
    """
    etapa = state.get("etapa_atual", "inicio")

    if etapa == "inicio":
        return "verificar_exames"
    if etapa == "exames_verificados":
        return "sugerir_conduta"
    if etapa in ("conduta_sugerida", "erro"):
        return "emitir_alertas"
    return "fim"  # "concluido" ou qualquer outro caso


def build_graph(knowledge_base, llm, db_path: str):
    verificar_exames = make_verificar_exames_node(db_path)
    sugerir_conduta = make_sugerir_conduta_node(knowledge_base, llm, db_path)

    workflow = StateGraph(AssistantState)

    workflow.add_node("verificar_exames", verificar_exames)
    workflow.add_node("sugerir_conduta", sugerir_conduta)
    workflow.add_node("emitir_alertas", emitir_alertas)

    workflow.set_entry_point("verificar_exames")

    # Arestas condicionais: o proximo no e decidido em runtime por _deve_continuar,
    # consultando o estado que cada no acabou de atualizar.
    workflow.add_conditional_edges(
        "verificar_exames", _deve_continuar, {"sugerir_conduta": "sugerir_conduta", "fim": END}
    )
    workflow.add_conditional_edges(
        "sugerir_conduta", _deve_continuar, {"emitir_alertas": "emitir_alertas", "fim": END}
    )
    workflow.add_conditional_edges("emitir_alertas", _deve_continuar, {"fim": END})

    # Checkpointer: persiste o estado por thread_id, permitindo retomar ou
    # auditar uma execucao especifica depois.
    memoria = MemorySaver()
    return workflow.compile(checkpointer=memoria)
