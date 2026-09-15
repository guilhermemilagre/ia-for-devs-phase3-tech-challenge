"""
Fine-tuning de LLM com dados medicos internos usando LoRA (PEFT).

Pensado para rodar em uma GPU unica (ex: Google Colab, T4/A100) usando
quantizacao de 4 bits (QLoRA) para reduzir uso de memoria.

Modelo base sugerido: um modelo aberto e leve, ex:
    - "meta-llama/Llama-3.2-1B-Instruct"
    - "microsoft/Phi-3-mini-4k-instruct"
    - "mistralai/Mistral-7B-Instruct-v0.3" (exige mais VRAM)

Uso:
    python src/finetune.py \
        --base-model meta-llama/Llama-3.2-1B-Instruct \
        --train-file data/processed/train.jsonl \
        --val-file data/processed/val.jsonl \
        --output-dir models/assistente-medico-lora \
        --epochs 3

Requisitos (requirements.txt):
    transformers>=4.44
    peft>=0.12
    bitsandbytes>=0.43
    accelerate>=0.33
    datasets>=2.20
    torch
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


PROMPT_TEMPLATE = (
    "### Instrucao:\n{instruction}\n\n"
    "### Contexto:\n{input}\n\n"
    "### Resposta:\n{output}"
)


def format_example(example: dict) -> dict:
    text = PROMPT_TEMPLATE.format(
        instruction=example["instruction"],
        input=example.get("input", ""),
        output=example["output"],
    )
    return {"text": text}


def tokenize_function(examples, tokenizer, max_length: int = 512):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--use-4bit", action="store_true", default=True)
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # 1. Carregar tokenizer e modelo base (quantizado em 4 bits - QLoRA)
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=args.use_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config if args.use_4bit else None,
        device_map="auto",
    )

    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # -----------------------------------------------------------------
    # 2. Configurar LoRA
    # -----------------------------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # -----------------------------------------------------------------
    # 3. Carregar e preparar dataset
    # -----------------------------------------------------------------
    dataset = load_dataset(
        "json",
        data_files={"train": str(args.train_file), "validation": str(args.val_file)},
    )
    dataset = dataset.map(format_example)
    dataset = dataset.map(
        lambda ex: tokenize_function(ex, tokenizer, args.max_length),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # -----------------------------------------------------------------
    # 4. Treinamento
    # -----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
    )

    trainer.train()

    # -----------------------------------------------------------------
    # 5. Salvar adaptador LoRA (nao o modelo base inteiro)
    # -----------------------------------------------------------------
    model.save_pretrained(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print(f"Adaptador LoRA salvo em: {args.output_dir}")


if __name__ == "__main__":
    main()
