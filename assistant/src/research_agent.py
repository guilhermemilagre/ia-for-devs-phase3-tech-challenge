"""
Agente de pesquisa livre - demonstra o padrao de selecao dinamica de
ferramentas (@tool + ToolNode/tools_condition, Aula 04 de LangGraph:
Multi-Agent) SEPARADO da decisao clinica principal.

Por que isolado do pipeline de conduta (graph.py/nodes.py): la, a seguranca
vem justamente do fluxo ser fixo (verificar_exames -> sugerir_conduta ->
emitir_alertas sempre roda, sem exceção). Num agente com selecao dinamica de
ferramentas, e o proprio LLM que decide se/quando chama cada ferramenta -
otimo para uma pergunta de pesquisa livre e generica, arriscado para decisao
de conduta de paciente (o LLM poderia "decidir" nao checar exames criticos).
Ver a discussao completa no README (secao de limitacoes).

Uso (CLI):
    python assistant/src/main.py --modo pesquisa \
        --pergunta "Qual o protocolo de sepse da instituicao?"

Ferramentas disponiveis ao agente:
    - buscar_protocolo(query): busca semantica nos protocolos institucionais (RAG)
    - consultar_paciente(paciente_id): dados cadastrais/clinicos de um paciente
"""

from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition


def _make_tools(knowledge_base, db_path: str):
    @tool
    def buscar_protocolo(query: str) -> str:
        """Busca nos protocolos institucionais (FAQs, protocolos, modelos de laudo)
        por informacao relevante a query. Use para perguntas gerais sobre
        condutas/protocolos, sem paciente especifico em mente."""
        resultados = knowledge_base.similarity_search_with_score(query, k=3)
        if not resultados:
            return "Nenhum protocolo relevante encontrado."
        return "\n\n".join(
            f"[Fonte: {doc.metadata.get('fonte', 'desconhecida')}]\n{doc.page_content}"
            for doc, _ in resultados
        )

    @tool
    def consultar_paciente(paciente_id: int) -> str:
        """Retorna diagnostico, medicacoes em uso e exames pendentes de um
        paciente pelo ID. Use quando a pergunta menciona um paciente especifico."""
        import sqlite3

        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "SELECT nome_iniciais, idade, sexo, diagnostico_principal "
            "FROM pacientes WHERE id = ?",
            (paciente_id,),
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            return f"Paciente {paciente_id} nao encontrado."
        nome, idade, sexo, diagnostico = row

        cur.execute(
            "SELECT nome_exame FROM exames WHERE paciente_id = ? AND status = 'pendente'",
            (paciente_id,),
        )
        pendentes = [r[0] for r in cur.fetchall()]
        conn.close()

        return (
            f"Paciente {nome}, {idade} anos, {sexo}. Diagnostico: {diagnostico}. "
            f"Exames pendentes: {', '.join(pendentes) or 'nenhum'}."
        )

    return [buscar_protocolo, consultar_paciente]


def build_research_graph(llm, knowledge_base, db_path: str):
    """Constroi o grafo do agente de pesquisa livre (ToolNode/tools_condition
    do langgraph.prebuilt - nao usado no pipeline clinico principal)."""
    tools = _make_tools(knowledge_base, db_path)
    llm_com_ferramentas = llm.bind_tools(tools)

    def agente(state: MessagesState):
        return {"messages": [llm_com_ferramentas.invoke(state["messages"])]}

    grafo = StateGraph(MessagesState)
    grafo.add_node("agente", agente)
    grafo.add_node("tools", ToolNode(tools))
    grafo.add_edge(START, "agente")
    grafo.add_conditional_edges("agente", tools_condition)
    grafo.add_edge("tools", "agente")
    return grafo.compile()


def executar_pesquisa(llm, knowledge_base, db_path: str, pergunta: str) -> str:
    app = build_research_graph(llm, knowledge_base, db_path)
    resultado = app.invoke({"messages": [("user", pergunta)]})
    return resultado["messages"][-1].content
