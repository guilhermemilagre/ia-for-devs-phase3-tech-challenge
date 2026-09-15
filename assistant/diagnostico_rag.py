"""
Diagnostico do RAG: mostra o que a busca vetorial recupera de verdade,
sem passar pelos filtros/cortes do filter_and_rank - para descobrir se o
problema esta na recuperacao (embedding nao acha o documento certo) ou
no ranking (acha, mas descarta/perde para outro).

Uso (na raiz do projeto, com o venv/ambiente ja ativado):
    python assistant/diagnostico_rag.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from knowledge_base import load_knowledge_base

INDEX_DIR = "data/db/faiss_index"

# A mesma query que o sugerir_conduta monta para o paciente 3
QUERY = (
    "Qual a conduta recomendada para este paciente com base no protocolo "
    "institucional? Diagnostico do paciente: Suspeita de Sepse. "
    "Exames pendentes: Lactato serico, Hemocultura."
)

print("Carregando indice FAISS...")
kb = load_knowledge_base(INDEX_DIR)

print(f"\nQuery usada:\n{QUERY}\n")
print("=" * 80)
print("TODOS os documentos do indice, ordenados por distancia a essa query")
print("(menor distancia = mais similar/relevante)")
print("=" * 80)

# k grande o suficiente para trazer o indice inteiro (ajuste se seu dataset
# tiver mais de 50 documentos)
resultados = kb.similarity_search_with_score(QUERY, k=50)

for i, (doc, distancia) in enumerate(resultados, start=1):
    fonte = doc.metadata.get("fonte", "desconhecida")
    tipo = doc.metadata.get("tipo", "desconhecido")
    marcador = " <<<< PROTOCOLO DE SEPSE" if "sepse" in fonte.lower() and tipo == "protocolo" else ""
    print(f"{i:2d}. distancia={distancia:.4f}  tipo={tipo:15s}  fonte={fonte}{marcador}")

print()
print("=" * 80)
print("Se o 'Protocolo de Sepse' aparecer longe do topo (ou nem aparecer),")
print("o problema esta na RECUPERACAO (embedding/indice), nao no ranking.")
print("Se ele aparecer perto do topo mas ainda assim nao vencer no")
print("resultado final do assistente, o problema esta no ranking/filtro.")
print("=" * 80)
