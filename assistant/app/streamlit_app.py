"""
Interface do assistente medico virtual - Streamlit consumindo a API FastAPI
(assistant/api/main.py), que por sua vez chama o modelo fine-tuned via
Ollama de verdade. Espelha o padrao API + app da Fase 2.

Uso (com a API rodando, ver assistant/api/main.py):
    streamlit run assistant/app/streamlit_app.py
"""

import os
import re

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

_FONTE_COM_URL = re.compile(r"^(?P<nome>.+?) \((?P<url>https?://\S+)\)$")


def formatar_fonte(fonte: str) -> str:
    """Documentos oficiais (PCDT/ILAS) trazem a URL real entre parenteses no
    final da fonte (ver rag_nodes.py::generate_context) - aqui viram um link
    clicavel de verdade, em vez de texto cru com a URL solta."""
    m = _FONTE_COM_URL.match(fonte)
    if m:
        return f"{m.group('nome')} — [abrir documento original]({m.group('url')})"
    return fonte

st.set_page_config(page_title="Assistente Medico Virtual", page_icon="🩺", layout="centered")

st.title("🩺 Assistente Médico Virtual")
st.caption(
    "Pipeline LangGraph (RAG + LLM fine-tuned via Ollama + guardrails). "
    "Sugestões sempre sujeitas a validação humana."
)


@st.cache_data(ttl=30)
def carregar_pacientes():
    resp = requests.get(f"{API_URL}/pacientes", timeout=10)
    resp.raise_for_status()
    return resp.json()["pacientes"]


def verificar_saude_api():
    try:
        resp = requests.get(f"{API_URL}/health", timeout=5)
        return resp.json()
    except requests.exceptions.RequestException as e:
        return {"status": "offline", "erro": str(e)}


saude = verificar_saude_api()
if saude.get("status") != "ok":
    st.error(
        f"⚠️ API indisponível em `{API_URL}` (status: {saude.get('status')}). "
        f"Rode: `uvicorn main:app --app-dir assistant/api --port 8000` "
        f"e confirme que o Ollama está rodando com o modelo `assistente-medico`."
    )
    st.caption(f"Detalhe: {saude.get('erro', '')}")
    st.stop()

st.success(f"✅ API conectada — modelo Ollama: `{saude.get('ollama_model')}`", icon="✅")

try:
    pacientes = carregar_pacientes()
except requests.exceptions.RequestException as e:
    st.error(f"Não foi possível carregar a lista de pacientes: {e}")
    st.stop()

opcoes = {
    f"Paciente {p['id']} — {p['nome_iniciais']} ({p['diagnostico_principal']})": p
    for p in pacientes
}
escolha = st.selectbox("Selecione o paciente:", list(opcoes.keys()))
paciente = opcoes[escolha]

col1, col2, col3 = st.columns(3)
col1.metric("Idade", paciente["idade"])
col2.metric("Sexo", paciente["sexo"])
col3.metric("Diagnóstico", paciente["diagnostico_principal"])

with st.expander("Exames do paciente"):
    for exame in paciente["exames"]:
        icone = "🔴" if exame["status"] == "pendente" else "✅"
        st.write(f"{icone} {exame['nome']} — {exame['status']}")

pergunta = st.text_area(
    "Pergunta ao assistente:",
    value="Qual a conduta recomendada para este paciente com base no protocolo institucional?",
)

if st.button("Consultar assistente", type="primary"):
    with st.spinner("Consultando protocolos e gerando sugestão (LLM via Ollama)..."):
        try:
            resp = requests.post(
                f"{API_URL}/consultar",
                json={"paciente_id": paciente["id"], "pergunta": pergunta},
                timeout=120,
            )
            resp.raise_for_status()
            resultado = resp.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao consultar a API: {e}")
            st.stop()

    for alerta in resultado.get("alertas", []):
        st.error(f"🚨 {alerta}")

    if resultado.get("confianca_baixa"):
        st.warning(
            f"⚠️ Confiança de recuperação baixa (confidence_score="
            f"{resultado['confidence_score']:.2f}) — revisar as fontes com atenção redobrada."
        )

    st.subheader("Conduta sugerida")
    st.write(resultado.get("conduta_sugerida", "<não gerada>"))

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Urgência", resultado.get("urgencia", "não classificado"))
    col_b.metric("Confiança (RAG)", f"{resultado.get('confidence_score', 0):.2f}")
    col_c.metric("Guardrail acionado", "Sim" if resultado.get("guardrail_acionado") else "Não")

    with st.expander(f"Fontes consultadas ({len(resultado.get('fontes', []))})"):
        for fonte in resultado.get("fontes", []):
            st.markdown(f"- {formatar_fonte(fonte)}")

    with st.expander("Histórico de decisões (trilha ReAct)"):
        for linha in resultado.get("historico_decisoes", []):
            st.write(f"- {linha}")

st.divider()
st.subheader("🔎 Modo pesquisa livre")
st.caption(
    "Agente com seleção dinâmica de ferramentas (@tool/ToolNode) — NÃO passa pelos "
    "guardrails do pipeline clínico. Use só para perguntas gerais, não para decisão sobre paciente."
)
pergunta_livre = st.text_input("Pergunta livre:", key="pergunta_livre")
if st.button("Perguntar"):
    with st.spinner("Consultando..."):
        try:
            resp = requests.post(
                f"{API_URL}/pesquisa", json={"pergunta": pergunta_livre}, timeout=120
            )
            resp.raise_for_status()
            st.write(resp.json()["resposta"])
        except requests.exceptions.RequestException as e:
            st.error(f"Erro ao consultar a API: {e}")
