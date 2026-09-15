#!/usr/bin/env python3
"""
Expande o dataset sintetico com FAQs reais do MedQuAD, filtradas pelo campo
<Focus> do XML para as 3 doencas do projeto (hipertensao, diabetes, sepse).

Diferente do PubMedQA (perguntas de pesquisa biomedica em geral, nao
filtraveis por doenca - ver expand_dataset.py), o MedQuAD tem um campo
<Focus> por documento que permite filtrar por doenca
real, dando conteudo genuinamente no dominio clinico do projeto. Confirmado
em 2026-09-13 escaneando o repositorio inteiro: 23 documentos de
hipertensao, 97 de diabetes, 8 de sepse (contagem bruta por substring no
Focus, antes da curadoria manual de FOCOS_PRIORITARIOS abaixo).

FOCOS_PRIORITARIOS foi escolhido a dedo (nao "pegar tudo") porque a maioria
dos 128 documentos brutos e sobre complicacoes cronicas/autocuidado (ex:
"Diabetes - cuidado com os pes", "Hipertensao pulmonar" - uma doenca
diferente que so compartilha a palavra "hipertensao") - fora do escopo de
"conduta clinica na internacao" que e o caso de uso deste projeto.

Uso:
    git clone --depth 1 https://github.com/abachaa/MedQuAD.git /caminho/MedQuAD
    python finetuning/src/expand_dataset_medquad.py \
        --medquad-dir /caminho/MedQuAD \
        --output data/raw/dataset_medquad.jsonl
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

FOCOS_PRIORITARIOS = {
    "hipertensao": [
        "High blood pressure",
        "Controlling your high blood pressure",
        "High blood pressure medicines",
        "High blood pressure and diet",
        "Malignant hypertension",
    ],
    "diabetes": [
        "Diabetes",
        "Diabetes and kidney disease",
        "Diabetes - when you are sick",
        "Diabetes - preventing heart attack and stroke",
    ],
    "sepse": [
        "Sepsis",
        "Septic shock",
        "Septicemia",
    ],
}

MAX_RESPOSTA_CHARS = 600


def coletar_registros(medquad_dir: Path) -> list[dict]:
    focos_alvo = {foco.lower() for focos in FOCOS_PRIORITARIOS.values() for foco in focos}

    registros = []
    for xml_path in sorted(medquad_dir.rglob("*.xml")):
        try:
            tree = ET.parse(xml_path)
        except ET.ParseError:
            continue

        focus_el = tree.find("Focus")
        focus = (focus_el.text or "").strip() if focus_el is not None else ""
        if focus.lower() not in focos_alvo:
            continue

        fonte_original = xml_path.parent.name
        for qa in tree.findall(".//QAPair"):
            pergunta_el = qa.find("Question")
            resposta_el = qa.find("Answer")
            if pergunta_el is None or resposta_el is None:
                continue
            pergunta = (pergunta_el.text or "").strip()
            resposta = (resposta_el.text or "").strip()
            if not pergunta or not resposta:
                continue

            if len(resposta) > MAX_RESPOSTA_CHARS:
                cortada = resposta[:MAX_RESPOSTA_CHARS].rsplit(".", 1)[0]
                resposta = (cortada or resposta[:MAX_RESPOSTA_CHARS]) + "."

            registros.append(
                {
                    "tipo": "faq",
                    "pergunta": pergunta,
                    "resposta": resposta,
                    "fonte": f"MedQuAD (NIH {fonte_original}, Focus: {focus})",
                }
            )
    return registros


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--medquad-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/dataset_medquad.jsonl"))
    args = parser.parse_args()

    registros = coletar_registros(args.medquad_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in registros:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{len(registros)} FAQs reais do MedQuAD salvas em {args.output}")
    focos_vistos = sorted({r["fonte"].split("Focus: ")[1].rstrip(")") for r in registros})
    print("Focos incluidos:", focos_vistos)


if __name__ == "__main__":
    main()
