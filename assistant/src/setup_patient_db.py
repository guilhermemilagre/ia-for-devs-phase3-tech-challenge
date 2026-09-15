"""
Cria uma base de dados SQLite simulando prontuários/registros de pacientes,
usada pelo assistente para contextualizar respostas com informações
atualizadas do paciente (ex: exames pendentes, diagnósticos, medicações em uso).

Uso:
    python src/setup_patient_db.py --db-path data/db/prontuarios.db
"""

import argparse
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    id INTEGER PRIMARY KEY,
    nome_iniciais TEXT NOT NULL,   -- apenas iniciais, dado real anonimizado
    idade INTEGER,
    sexo TEXT,
    diagnostico_principal TEXT
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY,
    paciente_id INTEGER,
    nome_exame TEXT,
    status TEXT,          -- 'pendente' | 'concluido'
    resultado TEXT,
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);

CREATE TABLE IF NOT EXISTS medicacoes_em_uso (
    id INTEGER PRIMARY KEY,
    paciente_id INTEGER,
    nome_medicacao TEXT,
    dose TEXT,
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);
"""

SEED_DATA = {
    "pacientes": [
        (1, "J.S.", 62, "M", "Hipertensao Arterial"),
        (2, "M.A.", 54, "F", "Diabetes Mellitus tipo 2"),
        (3, "R.O.", 71, "M", "Suspeita de Sepse"),
    ],
    "exames": [
        (1, 1, "Creatinina", "concluido", "1.1 mg/dL"),
        (2, 1, "Potassio", "pendente", None),
        (3, 1, "ECG", "concluido", "sem alteracoes agudas"),
        (4, 2, "HbA1c", "pendente", None),
        (5, 2, "Funcao renal (TFG)", "concluido", "TFG 68 mL/min"),
        (6, 3, "Lactato serico", "pendente", None),
        (7, 3, "Hemocultura", "pendente", None),
    ],
    "medicacoes_em_uso": [
        (1, 1, "Losartana", "50mg 1x/dia"),
        (2, 2, "Metformina", "500mg 2x/dia"),
    ],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, required=True)
    args = parser.parse_args()

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.db_path)
    cur = conn.cursor()
    cur.executescript(SCHEMA)

    cur.executemany(
        "INSERT OR REPLACE INTO pacientes VALUES (?, ?, ?, ?, ?)",
        SEED_DATA["pacientes"],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO exames VALUES (?, ?, ?, ?, ?)",
        SEED_DATA["exames"],
    )
    cur.executemany(
        "INSERT OR REPLACE INTO medicacoes_em_uso VALUES (?, ?, ?, ?)",
        SEED_DATA["medicacoes_em_uso"],
    )

    conn.commit()
    conn.close()
    print(f"Base de dados simulada criada em: {args.db_path}")


if __name__ == "__main__":
    main()
