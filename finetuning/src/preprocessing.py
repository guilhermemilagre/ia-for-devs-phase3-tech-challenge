"""
Preprocessing e anonimizacao do dataset medico interno.

Etapas:
1. Leitura do dataset bruto (JSONL) com protocolos, FAQs e modelos de laudo.
2. Anonimizacao de PII (nomes, CPF, datas, telefones) via regex.
3. Conversao para o formato de instruction-tuning:
   {"instruction": ..., "input": ..., "output": ...}
4. Curadoria basica: remocao de duplicatas e registros vazios.
5. Split em treino/validacao e escrita em data/processed/.

Uso:
    python src/preprocessing.py \
        --input data/raw/dataset_sintetico.jsonl \
        --output-dir data/processed \
        --val-size 0.15
"""

import argparse
import json
import re
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Anonimizacao
# ---------------------------------------------------------------------------

PATTERNS = {
    # CPF: 000.000.000-00 ou 00000000000
    "cpf": re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"),
    # Datas dd/mm/aaaa ou dd-mm-aaaa
    "data": re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b"),
    # Telefones (10 ou 11 digitos, com ou sem parenteses/hifen)
    "telefone": re.compile(r"\(?\d{2}\)?\s?\d{4,5}-?\d{4}\b"),
    # Placeholders de nome ja usados no dataset sintetico, ex: [nome]
    "campo_livre": re.compile(r"\[(nome|idade|descricao|tempo|medicacoes|valor|prazo)\]", re.IGNORECASE),
}


def anonymize(text: str) -> str:
    """Substitui CPF, datas e telefones por marcadores neutros.
    Mantem placeholders de template (ex: [nome]) intactos, pois já
    representam campos a serem preenchidos, não dados reais.
    """
    text = PATTERNS["cpf"].sub("[CPF_REDACTED]", text)
    text = PATTERNS["data"].sub("[DATA_REDACTED]", text)
    text = PATTERNS["telefone"].sub("[TELEFONE_REDACTED]", text)
    return text


# ---------------------------------------------------------------------------
# Conversao para formato instruction-tuning
# ---------------------------------------------------------------------------

def record_to_instruction(record: dict) -> dict:
    tipo = record.get("tipo")

    if tipo == "protocolo":
        instruction = (
            f"Explique a conduta clinica recomendada segundo o protocolo "
            f"institucional: {record['titulo']}"
        )
        return {
            "instruction": instruction,
            "input": "",
            "output": anonymize(record["conteudo"]),
            "fonte": record["titulo"],  # usado depois para explainability
        }

    if tipo == "faq":
        return {
            "instruction": anonymize(record["pergunta"]),
            "input": "",
            "output": anonymize(record["resposta"]),
            "fonte": "FAQ interna",
        }

    if tipo == "laudo_modelo":
        instruction = f"Gere um modelo de {record['titulo'].lower()}"
        return {
            "instruction": instruction,
            "input": "",
            "output": anonymize(record["conteudo"]),
            "fonte": record["titulo"],
        }

    raise ValueError(f"Tipo de registro desconhecido: {tipo}")


# ---------------------------------------------------------------------------
# Curadoria
# ---------------------------------------------------------------------------

def curate(records: list[dict]) -> list[dict]:
    seen = set()
    curated = []
    for r in records:
        key = (r["instruction"].strip().lower(), r["output"].strip().lower())
        if not r["instruction"].strip() or not r["output"].strip():
            continue  # descarta vazios
        if key in seen:
            continue  # descarta duplicatas
        seen.add(key)
        curated.append(r)
    return curated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_records.append(json.loads(line))

    converted = [record_to_instruction(r) for r in raw_records]
    curated = curate(converted)

    random.seed(args.seed)
    random.shuffle(curated)

    n_val = max(1, int(len(curated) * args.val_size))
    val_set = curated[:n_val]
    train_set = curated[n_val:]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    def write_jsonl(path: Path, data: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

    write_jsonl(args.output_dir / "train.jsonl", train_set)
    write_jsonl(args.output_dir / "val.jsonl", val_set)

    print(f"Registros brutos: {len(raw_records)}")
    print(f"Registros apos curadoria: {len(curated)}")
    print(f"Treino: {len(train_set)} | Validacao: {len(val_set)}")
    print(f"Arquivos salvos em: {args.output_dir}")


if __name__ == "__main__":
    main()
