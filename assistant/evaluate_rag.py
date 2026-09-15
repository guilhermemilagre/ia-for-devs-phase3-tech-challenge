#!/usr/bin/env python3
"""
Avaliacao quantitativa do RAG: precision, recall e F1 da recuperacao de
documentos, por paciente e agregado (macro-media).

Preenche o gap identificado na auditoria de cobertura das disciplinas da
Fase 3 (Aula 01 - Analise e Otimizacao de Prompts: metricas de avaliacao) -
ate 2026-09-13 a "avaliacao do modelo e analise dos resultados" exigida pelo
enunciado do Tech Challenge era so qualitativa (ler a resposta e achar que
fez sentido clinicamente).

Deliberadamente NAO chama o LLM (`filter_and_rank(..., llm=None)` - o
parametro so existe por compatibilidade de assinatura, ja nao e usado para
decidir relevancia, ver rag_nodes.py). Isso torna a avaliacao 100%
deterministica e reproduzivel, ao contrario da geracao de texto (que varia
com temperature > 0, ja documentado como limitacao no relatorio tecnico).

Metodologia (simples e deliberadamente objetiva, para nao exigir julgamento
humano por documento): um documento e "relevante" para um paciente se sua
`fonte` menciona a doenca do diagnostico principal dele
(GROUND_TRUTH_KEYWORD). Isso sub-conta relevancia real (uma FAQ do mesmo
dominio tambem ajudaria mesmo sem "vencer" como fonte principal, e o
checklist generico de alta hospitalar as vezes e legitimamente util) - e uma
limitacao conhecida do metodo, registrada aqui e no relatorio tecnico, nao
escondida.

Uso:
    python assistant/evaluate_rag.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import sqlite3

from knowledge_base import build_documents, load_knowledge_base
from rag_nodes import filter_and_rank, retrieve_documents
from states import RAGState

DB_PATH = "data/db/prontuarios.db"

# paciente_id -> palavra-chave que identifica documentos relevantes na fonte.
GROUND_TRUTH_KEYWORD = {
    1: "Hipertens",
    2: "Diabetes",
    3: "Sepse",
}


def _get_paciente(db_path: str, paciente_id: int) -> tuple[str, list[str]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT diagnostico_principal FROM pacientes WHERE id = ?", (paciente_id,))
    diagnostico = cur.fetchone()[0]
    cur.execute(
        "SELECT nome_exame FROM exames WHERE paciente_id = ? AND status = 'pendente'",
        (paciente_id,),
    )
    pendentes = [r[0] for r in cur.fetchall()]
    conn.close()
    return diagnostico, pendentes


def _relevante(fonte: str, keyword: str) -> bool:
    return keyword.lower() in fonte.lower()


def avaliar_paciente(kb, db_path: str, paciente_id: int, todos_documentos) -> dict:
    diagnostico, exames_pendentes = _get_paciente(db_path, paciente_id)
    keyword = GROUND_TRUTH_KEYWORD[paciente_id]

    # Mesma logica de enriquecimento de query usada em nodes.py
    # (make_sugerir_conduta_node) - a avaliacao precisa exercitar o mesmo
    # caminho que o pipeline real usa, senao os numeros nao significam nada.
    query = "Qual a conduta recomendada para este paciente com base no protocolo institucional?"
    partes = [f"Diagnostico do paciente: {diagnostico}."]
    if exames_pendentes:
        partes.append(f"Exames pendentes: {', '.join(exames_pendentes)}.")
    query_enriquecida = f"{query} {' '.join(partes)}"

    rag_state: RAGState = {"pergunta": query_enriquecida}
    rag_state = retrieve_documents(rag_state, kb)
    rag_state = filter_and_rank(rag_state, llm=None)

    recuperados = rag_state["filtered_documents"]
    relevantes_recuperados = [d for d in recuperados if _relevante(d["fonte"], keyword)]

    total_relevantes_corpus = sum(
        1 for doc in todos_documentos if _relevante(doc.metadata["fonte"], keyword)
    )

    precisao = len(relevantes_recuperados) / len(recuperados) if recuperados else 0.0
    recall = (
        len(relevantes_recuperados) / total_relevantes_corpus if total_relevantes_corpus else 0.0
    )
    f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0

    return {
        "paciente_id": paciente_id,
        "diagnostico": diagnostico,
        "confidence_score": rag_state.get("confidence_score", 0.0),
        "recuperados": len(recuperados),
        "relevantes_recuperados": len(relevantes_recuperados),
        "total_relevantes_corpus": total_relevantes_corpus,
        "precisao": precisao,
        "recall": recall,
        "f1": f1,
        # hit@k: pelo menos um documento relevante chegou ao contexto final?
        # Metrica complementar ao recall "classico" (relevantes recuperados /
        # total relevantes no corpus inteiro) - que degrada mecanicamente
        # conforme o corpus cresce (um top_k fixo nunca vai "recall=1.0" se
        # existirem 22 documentos relevantes), mesmo quando o sistema
        # continua entregando o essencial. hit@k e o que importa na pratica
        # aqui: o medico so precisa de UM bom protocolo no contexto, nao de
        # todos os documentos relevantes existentes.
        "hit_top_k": len(relevantes_recuperados) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="data/db/faiss_index")
    parser.add_argument("--dataset", default="data/raw/dataset_sintetico.jsonl")
    args = parser.parse_args()

    kb = load_knowledge_base(args.index_dir)
    todos_documentos = build_documents(Path(args.dataset))

    resultados = [
        avaliar_paciente(kb, DB_PATH, pid, todos_documentos) for pid in GROUND_TRUTH_KEYWORD
    ]

    print("=" * 92)
    print(f"AVALIACAO QUANTITATIVA DO RAG (precision / recall / F1) - indice: {args.index_dir}")
    print("=" * 92)
    print(f"{'Paciente':<10}{'Diagnostico':<27}{'Confidence':<13}{'Precisao':<11}{'Recall':<11}{'F1':<10}{'Hit@k':<8}")
    print("-" * 92)
    for r in resultados:
        print(
            f"{r['paciente_id']:<10}{r['diagnostico']:<27}{r['confidence_score']:<13.2f}"
            f"{r['precisao']:<11.2f}{r['recall']:<11.2f}{r['f1']:<10.2f}{str(r['hit_top_k']):<8}"
        )

    n = len(resultados)
    macro_precisao = sum(r["precisao"] for r in resultados) / n
    macro_recall = sum(r["recall"] for r in resultados) / n
    macro_f1 = sum(r["f1"] for r in resultados) / n
    hit_rate = sum(r["hit_top_k"] for r in resultados) / n
    print("-" * 92)
    print(
        f"{'MACRO AVG':<10}{'':<27}{'':<13}{macro_precisao:<11.2f}{macro_recall:<11.2f}"
        f"{macro_f1:<10.2f}{hit_rate:<8.2f}"
    )
    print()
    print("Detalhe por paciente (recuperados / relevantes no top-k / relevantes no corpus):")
    for r in resultados:
        print(
            f"  Paciente {r['paciente_id']}: {r['recuperados']} recuperados, "
            f"{r['relevantes_recuperados']} relevantes no top-k, "
            f"{r['total_relevantes_corpus']} relevantes existentes no corpus inteiro."
        )
    print()
    print("Metodologia: um documento e 'relevante' se sua fonte menciona a doenca do")
    print("diagnostico principal do paciente (ver GROUND_TRUTH_KEYWORD no topo do arquivo).")
    print("Precisao = relevantes recuperados / total recuperados no top-k.")
    print("Recall = relevantes recuperados / total de documentos relevantes no corpus inteiro.")


if __name__ == "__main__":
    main()
