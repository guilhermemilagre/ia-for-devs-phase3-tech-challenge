"""
Script principal do assistente medico virtual.

Uso:
    python src/main.py --paciente-id 3 \
        --pergunta "Qual a conduta recomendada para este paciente?" \
        --base-model meta-llama/Llama-3.2-1B-Instruct

Fluxo:
    1. Carrega a base vetorial de protocolos (FAISS) e o modelo (base + LoRA).
    2. Constroi o grafo com arestas condicionais e checkpointer (src/graph.py).
    3. Executa o grafo para o paciente/pergunta informados, usando um
       thread_id para permitir retomar/auditar a execucao depois.
    4. Imprime exames pendentes, conduta sugerida, fontes, alertas e o
       historico de decisoes (trilha ReAct) - alem da estrutura ASCII do
       grafo, como ensinado nas aulas de LangGraph.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from knowledge_base import load_knowledge_base
from graph import build_graph
from states import criar_estado_inicial


def carregar_llm_hf(base_model: str, lora_adapter_path: str | None, max_new_tokens: int = 300):
    """Carrega o modelo via Transformers + PEFT (requer GPU CUDA - Colab/RunPod/etc)."""
    try:
        from langchain_huggingface import HuggingFacePipeline
    except ImportError:
        from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    if lora_adapter_path:
        model = PeftModel.from_pretrained(model, lora_adapter_path)

    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
    )
    return HuggingFacePipeline(pipeline=text_gen_pipeline)


def carregar_llm_ollama(model_name: str, temperature: float = 0.2):
    """Carrega o modelo via Ollama local (Mac/Linux/Windows, sem GPU CUDA).

    Pressupoe que o servidor Ollama esta rodando (`ollama serve`) e que o
    modelo ja foi baixado (`ollama pull <model_name>`). O adaptador LoRA,
    quando existir, precisa ter sido fundido ao modelo base e convertido
    para GGUF antes (fora do escopo deste script) - o Ollama nao carrega
    adaptadores PEFT em tempo de execucao como o backend Hugging Face faz.
    """
    from langchain_ollama import ChatOllama

    return ChatOllama(model=model_name, temperature=temperature)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modo", choices=["conduta", "pesquisa"], default="conduta",
        help="conduta: pipeline clinico fixo (padrao, requer --paciente-id). "
             "pesquisa: agente de pesquisa livre com selecao dinamica de "
             "ferramentas (@tool/ToolNode) - NAO usado para decisao clinica, "
             "so para perguntas gerais/exploratorias. Ver assistant/src/research_agent.py.",
    )
    parser.add_argument("--paciente-id", type=int, help="Obrigatorio no modo 'conduta'")
    parser.add_argument(
        "--pergunta", type=str,
        default="Qual a conduta recomendada para este paciente com base no protocolo institucional?",
    )
    parser.add_argument(
        "--backend", choices=["huggingface", "ollama"], default="huggingface",
        help="huggingface: modelo base+LoRA via Transformers (requer GPU CUDA). "
             "ollama: modelo local via Ollama (Mac/sem GPU CUDA).",
    )
    parser.add_argument("--base-model", type=str, help="Necessario para --backend huggingface")
    parser.add_argument("--lora-adapter", type=str, default=None)
    parser.add_argument(
        "--ollama-model", type=str, default="llama3.2:1b",
        help="Nome do modelo ja baixado no Ollama (--backend ollama)",
    )
    parser.add_argument("--index-dir", type=str, default="data/db/faiss_index")
    parser.add_argument("--db-path", type=str, default="data/db/prontuarios.db")
    parser.add_argument("--show-graph", action="store_true", help="Imprime a estrutura ASCII do grafo")
    args = parser.parse_args()

    print("Carregando base de conhecimento (FAISS)...")
    index_path = Path(args.index_dir)
    if not (index_path / "index.faiss").exists():
        cwd = Path.cwd()
        parser.error(
            f"Indice FAISS nao encontrado em '{index_path}' (diretorio atual: {cwd}).\n"
            f"Isso costuma acontecer quando o projeto foi extraido/rodado a partir de "
            f"pastas diferentes entre uma celula e outra (ex: pasta duplicada apos "
            f"reiniciar a sessao e reextrair o zip). Confirme com `!pwd` e `!ls {index_path.parent}` "
            f"que voce esta na pasta certa, e rode de novo `python src/knowledge_base.py "
            f"--data data/raw/dataset_sintetico.jsonl --index-dir {index_path}` se necessario."
        )

    knowledge_base = load_knowledge_base(args.index_dir)

    print(f"Carregando modelo (backend: {args.backend})...")
    if args.backend == "ollama":
        llm = carregar_llm_ollama(args.ollama_model)
    else:
        if not args.base_model:
            parser.error("--base-model e obrigatorio quando --backend huggingface")
        llm = carregar_llm_hf(args.base_model, args.lora_adapter)

    if args.modo == "pesquisa":
        from research_agent import executar_pesquisa

        print("\n=== MODO PESQUISA LIVRE (agente com selecao dinamica de ferramentas) ===")
        print(
            "Aviso: este modo NAO passa pelos guardrails/alertas do pipeline clinico "
            "principal - use so para perguntas gerais, nao para decisao sobre paciente."
        )
        resposta = executar_pesquisa(llm, knowledge_base, args.db_path, args.pergunta)
        print(f"\nPergunta: {args.pergunta}")
        print(f"\nResposta: {resposta}")
        return

    if args.paciente_id is None:
        parser.error("--paciente-id e obrigatorio no modo 'conduta'")

    print("Construindo grafo do assistente...")
    app = build_graph(knowledge_base, llm, args.db_path)

    if args.show_graph:
        print("\n=== ESTRUTURA DO GRAFO ===")
        print(app.get_graph().draw_ascii())

    estado_inicial = criar_estado_inicial(args.paciente_id, args.pergunta)

    # thread_id identifica esta execucao para o checkpointer (MemorySaver) -
    # permite retomar ou inspecionar o estado depois pelo mesmo id.
    config = {"configurable": {"thread_id": f"paciente-{args.paciente_id}"}}
    resultado = app.invoke(estado_inicial, config=config)

    print("\n=== EXAMES PENDENTES ===")
    print(resultado.get("exames_pendentes") or "Nenhum")

    print("\n=== CONDUTA SUGERIDA ===")
    print(resultado.get("conduta_sugerida", "<nao gerada>"))

    print("\n=== FONTES ===")
    print(resultado.get("fontes", []))

    print("\n=== URGENCIA / CONFIANCA ===")
    rag_result = resultado.get("rag", {}) or {}
    print(f"Urgencia: {resultado.get('urgencia', 'nao classificado')} | "
          f"confidence_score: {rag_result.get('confidence_score', 0):.2f} | "
          f"confianca_baixa: {resultado.get('confianca_baixa', False)}")

    print("\n=== ALERTAS ===")
    print(resultado.get("alertas") or "Nenhum")

    print("\n=== HISTORICO DE DECISOES (ReAct) ===")
    for linha in resultado.get("historico_decisoes", []):
        print(f"- {linha}")


if __name__ == "__main__":
    main()
