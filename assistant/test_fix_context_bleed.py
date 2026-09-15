#!/usr/bin/env python3
"""
Teste de regressao do RAG: garante que cada paciente recupera o protocolo
certo (sem context bleed de outra doenca), sem chamar o LLM.

Reescrito em 2026-09-13. A versao anterior deste arquivo so chamava
`retrieve_documents` isolado (sem o enriquecimento com exames pendentes nem
`filter_and_rank`) e reportava "PASSOU" mesmo quando o proprio print dizia
"Documento de Sepse NAO foi recuperado" - testava uma fatia pequena e
desatualizada do pipeline (a estrategia de retrieval ja tinha mudado de MMR
para distancia pura, e a query ja tinha ganho o enriquecimento com exames
pendentes, sem que o teste acompanhasse). Corrigido para:
1. Usar `construir_query_enriquecida` de `nodes.py` - a MESMA funcao usada
   pelo pipeline real, nao uma copia manual que pode ficar desatualizada de
   novo.
2. Rodar `retrieve_documents` + `filter_and_rank` (as duas etapas reais),
   nao so a primeira.
3. Testar os 3 pacientes (nao so o de sepse), com asserção real: o teste
   FALHA (retorna 1) se o protocolo esperado nao estiver entre os
   documentos filtrados, em vez de so avisar e seguir "PASSOU" de qualquer
   jeito.

Sem GPU/LLM - so testa retrieve + filter_and_rank. Nao valida a geracao de
texto do LLM (essa parte e nao-deterministica mesmo com o contexto certo e
precisa de execucoes repetidas com o modelo real pra ser avaliada, o que
este teste rapido nao faz).

Uso:
    python assistant/test_fix_context_bleed.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import sqlite3

from knowledge_base import load_knowledge_base
from nodes import construir_query_enriquecida
from rag_nodes import filter_and_rank, retrieve_documents

DB_PATH = "data/db/prontuarios.db"
FAISS_INDEX = "data/db/faiss_index"

# paciente_id -> palavra-chave que deve aparecer na fonte de pelo menos um
# documento filtrado, para considerarmos que o protocolo certo foi
# recuperado (mesmo criterio usado em assistant/evaluate_rag.py).
CASOS_DE_TESTE = {
    1: "Hipertens",
    2: "Diabetes",
    3: "Sepse",
}


def _get_paciente(db_path: str, paciente_id: int) -> tuple[str, list[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT diagnostico_principal FROM pacientes WHERE id = ?", (paciente_id,))
    row = cur.fetchone()
    diagnostico = row[0] if row else None
    cur.execute(
        "SELECT nome_exame FROM exames WHERE paciente_id = ? AND status = 'pendente'",
        (paciente_id,),
    )
    pendentes = [r[0] for r in cur.fetchall()]
    conn.close()
    return diagnostico, pendentes


def testar_paciente(kb, paciente_id: int, keyword: str) -> bool:
    print("\n" + "=" * 70)
    print(f"Paciente {paciente_id} - esperado: fonte contendo '{keyword}'")
    print("=" * 70)

    diagnostico, exames_pendentes = _get_paciente(DB_PATH, paciente_id)
    if not diagnostico:
        print(f"❌ Paciente {paciente_id} não encontrado na base.")
        return False

    query = construir_query_enriquecida(
        "Qual a conduta recomendada para este paciente com base no protocolo institucional?",
        diagnostico,
        exames_pendentes,
    )
    print(f"Diagnóstico: {diagnostico} | Exames pendentes: {exames_pendentes or 'nenhum'}")
    print(f"Query enriquecida: {query}")

    rag_state = retrieve_documents({"pergunta": query}, kb)
    rag_state = filter_and_rank(rag_state, llm=None)
    filtrados = rag_state.get("filtered_documents", [])

    if not filtrados:
        print("❌ Nenhum documento passou pelo filtro (confidence_score=0.0).")
        return False

    print(f"confidence_score: {rag_state.get('confidence_score', 0):.2f}")
    print("Documentos filtrados:")
    encontrou = False
    for doc in filtrados:
        marcador = ""
        if keyword.lower() in doc["fonte"].lower():
            encontrou = True
            marcador = "  <<<< esperado"
        print(f"  - dist={doc['distancia']:.4f} fonte={doc['fonte'][:70]}{marcador}")

    if encontrou:
        print(f"\n✓ Documento sobre '{keyword}' recuperado corretamente.")
    else:
        print(f"\n❌ NENHUM documento sobre '{keyword}' entre os filtrados - possível context bleed.")

    return encontrou


def testar_fonte_faq_melhorada(kb) -> bool:
    """Confirma que fontes de FAQ incluem um trecho da pergunta (explainability),
    nao so 'FAQ interna' generico."""
    print("\n" + "=" * 70)
    print("Fonte de FAQ com trecho da pergunta (explainability)")
    print("=" * 70)

    rag_state = retrieve_documents({"pergunta": "Quais exames devo solicitar para hipertensao?"}, kb)
    fontes_faq = [d["fonte"] for d in rag_state["raw_documents"] if d["tipo"] == "faq"]

    if not fontes_faq:
        print("⚠️  Nenhuma FAQ recuperada nesta query - não é possível validar o formato da fonte.")
        return True  # não é uma falha do fix, so falta de dado pra essa query especifica

    genericas = [f for f in fontes_faq if f.strip() == "FAQ interna"]
    if genericas:
        print(f"❌ {len(genericas)} fonte(s) de FAQ ainda genérica(s) ('FAQ interna' sem detalhe).")
        return False

    print(f"✓ {len(fontes_faq)} fonte(s) de FAQ, todas com trecho específico:")
    for f in fontes_faq:
        print(f"  - {f[:80]}")
    return True


def main():
    faiss_path = Path(FAISS_INDEX)
    if not (faiss_path / "index.faiss").exists():
        print(f"❌ Índice FAISS não encontrado em {faiss_path}")
        print(f"   Execute primeiro: python assistant/src/knowledge_base.py --data data/raw/dataset_sintetico.jsonl --index-dir {FAISS_INDEX}")
        return 1

    kb = load_knowledge_base(FAISS_INDEX)

    resultados = {pid: testar_paciente(kb, pid, kw) for pid, kw in CASOS_DE_TESTE.items()}
    resultados["fonte_faq"] = testar_fonte_faq_melhorada(kb)

    print("\n" + "=" * 70)
    print("RESULTADO")
    print("=" * 70)
    for nome, ok in resultados.items():
        print(f"  {'✓ PASSOU' if ok else '❌ FALHOU'} - {nome}")

    if all(resultados.values()):
        print("\n✓ Todos os testes passaram - RAG recupera o protocolo certo para os 3 pacientes.")
        return 0

    print("\n❌ Um ou mais testes falharam - revisar acima antes de confiar no RAG.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
