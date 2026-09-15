"""
Guardrails e logging do assistente médico.

Guardrails:
- Bloqueia qualquer resposta que pareça uma prescricao direta e definitiva
  sem menção a validação humana.
- Garante que toda resposta inclua a fonte usada (explainability).

Logging:
- Cada interação é registrada em JSON Lines (logs/auditoria.jsonl) com:
  timestamp, pergunta, paciente_id (se houver), fontes usadas, resposta,
  e se algum guardrail foi acionado.
"""

import json
import re
import logging
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("logs/auditoria.jsonl")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Padrões que indicam prescrição direta / definitiva sem ressalva humana
PRESCRICAO_DIRETA_PATTERNS = [
    r"\btome\b.*\bagora\b",
    r"\baumente a dose\b(?!.*(m[eé]dico|valida[cç][aã]o|supervis))",
    r"\bpare de tomar\b(?!.*(m[eé]dico|valida[cç][aã]o))",
    r"\bestá autorizado a\b",
]

DISCLAIMER = (
    "\n\n[Aviso: sugestão gerada por IA. Requer validação e decisão final "
    "de um médico responsável antes de qualquer conduta.]"
)


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("assistente_medico")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


logger = _setup_logger()


def check_guardrails(resposta: str) -> tuple[str, bool]:
    """Verifica se a resposta viola os limites de atuação definidos.
    Retorna (resposta_ajustada, guardrail_acionado).
    """
    acionado = False
    for pattern in PRESCRICAO_DIRETA_PATTERNS:
        if re.search(pattern, resposta, flags=re.IGNORECASE):
            acionado = True
            break

    if acionado or DISCLAIMER not in resposta:
        resposta = resposta.rstrip() + DISCLAIMER

    return resposta, acionado


def log_interacao(
    pergunta: str,
    resposta: str,
    fontes: list[str],
    paciente_id: int | None = None,
    guardrail_acionado: bool = False,
    confidence_score: float | None = None,
    urgencia: str | None = None,
):
    """Registra a interação em JSON Lines para auditoria e rastreabilidade."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paciente_id": paciente_id,
        "pergunta": pergunta,
        "resposta": resposta,
        "fontes": fontes,
        "guardrail_acionado": guardrail_acionado,
        "confidence_score": confidence_score,
        "urgencia": urgencia,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    nivel = logging.WARNING if guardrail_acionado else logging.INFO
    logger.log(
        nivel,
        f"paciente_id={paciente_id} guardrail_acionado={guardrail_acionado} "
        f"fontes={fontes} pergunta='{pergunta[:60]}...'",
    )
