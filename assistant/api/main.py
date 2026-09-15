"""
API REST do assistente medico virtual - FastAPI envolvendo o mesmo pipeline
usado pelo CLI (assistant/src/main.py), sem reimplementar nada da logica
clinica. Espelha o padrao usado na Fase 2 (FastAPI + Streamlit), agora
falando de verdade com o modelo fine-tuned servido pelo Ollama local.

Pressupoe que o servidor Ollama esta rodando e que o modelo
`assistente-medico` (ou o que for passado em OLLAMA_MODEL) ja foi importado
(`ollama create assistente-medico -f finetuning/Modelfile`).

Uso (a partir da raiz do projeto):
    uvicorn main:app --reload --port 8000 --app-dir assistant/api

Depois, documentacao interativa em http://localhost:8000/docs
"""

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

_SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC_DIR))

from graph import build_graph  # noqa: E402
from knowledge_base import load_knowledge_base  # noqa: E402
from research_agent import executar_pesquisa  # noqa: E402
from states import criar_estado_inicial  # noqa: E402
import sqlite3  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402


def carregar_llm_ollama(model_name: str, temperature: float = 0.2) -> ChatOllama:
    """Mesma logica de assistant/src/main.py::carregar_llm_ollama, reimplementada
    aqui (nao importada de la) para evitar colisao de nome entre este arquivo
    (assistant/api/main.py) e assistant/src/main.py - os dois se chamam
    "main.py", e importar um "main" generico via sys.path seria fragil."""
    return ChatOllama(model=model_name, temperature=temperature)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "assistente-medico")
INDEX_DIR = os.environ.get("INDEX_DIR", "data/db/faiss_index_completo")
DB_PATH = os.environ.get("DB_PATH", "data/db/prontuarios.db")

app = FastAPI(
    title="Assistente Medico Virtual - API",
    description=(
        "Pipeline LangGraph (RAG + LLM fine-tuned via Ollama + guardrails) "
        "para apoio a decisao clinica. Sugestoes SEMPRE sujeitas a validacao "
        "humana - ver disclaimer em cada resposta."
    ),
    version="1.0.0",
)

# Estado carregado uma unica vez no startup (nao a cada requisicao) - FAISS,
# LLM e o grafo compilado sao reutilizaveis entre chamadas.
_estado = {}


@app.on_event("startup")
def carregar_recursos():
    if not (Path(INDEX_DIR) / "index.faiss").exists():
        raise RuntimeError(
            f"Indice FAISS nao encontrado em '{INDEX_DIR}'. Rode "
            f"assistant/src/knowledge_base.py primeiro, ou ajuste a env var INDEX_DIR."
        )
    _estado["knowledge_base"] = load_knowledge_base(INDEX_DIR)
    _estado["llm"] = carregar_llm_ollama(OLLAMA_MODEL)
    _estado["grafo"] = build_graph(_estado["knowledge_base"], _estado["llm"], DB_PATH)


class ConsultaRequest(BaseModel):
    paciente_id: int
    pergunta: str = "Qual a conduta recomendada para este paciente com base no protocolo institucional?"


class PesquisaRequest(BaseModel):
    pergunta: str


@app.get("/health")
def health():
    """Confirma que o indice FAISS e o modelo Ollama estao carregados e respondendo."""
    if "llm" not in _estado:
        raise HTTPException(status_code=503, detail="Recursos ainda nao carregados.")
    try:
        # Chamada minima real ao Ollama - confirma que o servidor esta de pe
        # e o modelo responde, nao so que o objeto ChatOllama foi criado.
        resposta = _estado["llm"].invoke("responda apenas: ok")
        ollama_ok = bool(resposta)
    except Exception as e:
        return {"status": "degradado", "ollama_ok": False, "erro": str(e)}
    return {
        "status": "ok",
        "ollama_ok": ollama_ok,
        "ollama_model": OLLAMA_MODEL,
        "index_dir": INDEX_DIR,
    }


@app.get("/pacientes")
def listar_pacientes():
    """Lista os pacientes simulados (id, diagnostico, exames pendentes) para
    preencher o seletor da interface."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id, nome_iniciais, idade, sexo, diagnostico_principal FROM pacientes")
    pacientes = []
    for pid, nome, idade, sexo, diagnostico in cur.fetchall():
        cur.execute(
            "SELECT nome_exame, status FROM exames WHERE paciente_id = ?", (pid,)
        )
        exames = [{"nome": n, "status": s} for n, s in cur.fetchall()]
        pacientes.append(
            {
                "id": pid,
                "nome_iniciais": nome,
                "idade": idade,
                "sexo": sexo,
                "diagnostico_principal": diagnostico,
                "exames": exames,
            }
        )
    conn.close()
    return {"pacientes": pacientes}


@app.post("/consultar")
def consultar(req: ConsultaRequest):
    """Executa o pipeline clinico completo (verificar exames -> sugerir
    conduta -> emitir alertas) para um paciente - chama o LLM via Ollama de
    verdade, sem mock."""
    if "grafo" not in _estado:
        raise HTTPException(status_code=503, detail="Recursos ainda nao carregados.")

    estado_inicial = criar_estado_inicial(req.paciente_id, req.pergunta)
    config = {"configurable": {"thread_id": f"paciente-{req.paciente_id}"}}
    try:
        resultado = _estado["grafo"].invoke(estado_inicial, config=config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao executar o pipeline: {e}")

    rag = resultado.get("rag", {}) or {}
    return {
        "paciente_id": req.paciente_id,
        "exames_pendentes": resultado.get("exames_pendentes", []),
        "exames_criticos_pendentes": resultado.get("exames_criticos_pendentes", []),
        "conduta_sugerida": resultado.get("conduta_sugerida", ""),
        "fontes": resultado.get("fontes", []),
        "urgencia": resultado.get("urgencia", "nao classificado"),
        "confidence_score": rag.get("confidence_score", 0.0),
        "confianca_baixa": resultado.get("confianca_baixa", False),
        "guardrail_acionado": resultado.get("guardrail_acionado", False),
        "alertas": resultado.get("alertas", []),
        "historico_decisoes": resultado.get("historico_decisoes", []),
    }


@app.post("/pesquisa")
def pesquisa(req: PesquisaRequest):
    """Modo de pesquisa livre (agente com selecao dinamica de ferramentas,
    ver assistant/src/research_agent.py) - NAO passa pelos guardrails do
    pipeline clinico, so para perguntas gerais/exploratorias."""
    if "grafo" not in _estado:
        raise HTTPException(status_code=503, detail="Recursos ainda nao carregados.")
    try:
        resposta = executar_pesquisa(
            _estado["llm"], _estado["knowledge_base"], DB_PATH, req.pergunta
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no agente de pesquisa: {e}")
    return {"pergunta": req.pergunta, "resposta": resposta}
