"""
Base de conhecimento vetorial (RAG) construída a partir dos protocolos
médicos internos. Usada pelo LangChain para contextualizar as respostas
do assistente com informações institucionais atualizadas — e para
permitir *explainability*, indicando qual protocolo embasou a resposta.

Uso como script (constrói e salva o índice):
    python src/knowledge_base.py --data data/raw/dataset_sintetico.jsonl \
        --index-dir data/db/faiss_index

Uso como módulo:
    from knowledge_base import load_knowledge_base
    kb = load_knowledge_base("data/db/faiss_index")
    kb.similarity_search_with_score("dose de metformina", k=2)
"""

import argparse
import json
from pathlib import Path

from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

try:
    # Pacote novo e recomendado (langchain >= 0.2). Requer `pip install langchain-huggingface`.
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Fallback para quem ainda nao instalou o pacote novo - funciona, mas com
    # o aviso de depreciacao do langchain-community.
    from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _carregar_registros_brutos(dataset_path: Path) -> list[dict]:
    """Carrega os registros brutos do JSONL via JSONLoader (Document Loaders,
    Aula 02 de LangChain na Pratica) em vez de abrir o arquivo e fazer
    json.loads linha a linha na mao.

    `text_content=False` porque cada linha e um dict heterogeneo (protocolo/
    faq/laudo_modelo tem campos diferentes) - nao ha uma unica string de
    conteudo para extrair aqui ainda. O JSONLoader serializa cada entrada em
    `page_content` (via json.dumps); a transformacao especifica por tipo
    (title+conteudo vs pergunta+resposta) continua em `build_documents`,
    porque isso e logica de dominio do projeto, nao parsing generico de
    arquivo - um Document Loader generico nao sabe que "faq" e "protocolo"
    devem virar texto diferente.
    """
    loader = JSONLoader(
        file_path=str(dataset_path),
        jq_schema=".",
        json_lines=True,
        text_content=False,
    )
    return [json.loads(doc.page_content) for doc in loader.load()]


def build_documents(dataset_path: Path) -> list[Document]:
    """Converte cada protocolo/FAQ/laudo em um Document com metadado de fonte,
    o que permite ao assistente citar de onde veio a informacao (explainability).
    """
    documents = []
    for record in _carregar_registros_brutos(dataset_path):
        tipo = record["tipo"]

        if tipo == "protocolo":
            content = f"{record['titulo']}\n{record['conteudo']}"
            fonte = record["titulo"]
        elif tipo == "faq":
            content = f"Pergunta: {record['pergunta']}\nResposta: {record['resposta']}"
            # Melhoria: incluir trecho da pergunta na fonte para melhor explainability
            pergunta_trecho = record['pergunta'][:60]
            fonte = f'FAQ interna: "{pergunta_trecho}..."'
        elif tipo == "laudo_modelo":
            content = f"{record['titulo']}\n{record['conteudo']}"
            fonte = record["titulo"]
        else:
            continue

        documents.append(
            Document(
                page_content=content,
                metadata={
                    "fonte": fonte,
                    "fonte_url": record.get("fonte_url", ""),
                    "tipo": tipo,
                    "oficial": bool(record.get("oficial", False)),
                    "doenca": record.get("doenca", ""),
                },
            )
        )
    return documents


def build_and_save_index(dataset_path: Path, index_dir: Path):
    documents = build_documents(dataset_path)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents, embeddings)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    print(f"Indice FAISS salvo em {index_dir} com {len(documents)} documentos.")


def load_knowledge_base(index_dir: str) -> FAISS:
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    args = parser.parse_args()
    build_and_save_index(args.data, args.index_dir)
