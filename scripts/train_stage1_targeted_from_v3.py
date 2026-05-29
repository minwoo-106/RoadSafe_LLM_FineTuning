import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft import PeftModel, prepare_model_for_kbit_training


MODEL_ID = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = Path("outputs/roadsafe_stage1_v3_clean_qlora/adapter")

TRAIN_PATH = Path("data/processed/roadsafe_stage1_targeted_mix_v1.jsonl")
VALID_PATH = Path("data/processed/stage1_v3_clean_split/valid.jsonl")

OUT_DIR = Path("outputs/roadsafe_stage1_targeted_from_v3_qlora")
MAX_LENGTH = 768


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_prompt(row):
    return (
        f"{row['instruction']}\n\n"
        f"상황: {row['input']}\n\n"
        f"답변:\n"
    )


class AnswerOnlyDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=768):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = build_prompt(row)
        answer = row["output"] + self.tokenizer.eos_token
        full_text = prompt + answer

        full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_ids = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )["input_ids"][0]

        input_ids = full["input_ids"][0]
        attention_mask = full["attention_mask"][0]
        labels = input_ids.clone()

        prompt_len = min(len(prompt_ids), self.max_length)
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main():
    print("[INFO] base model:", MODEL_ID)
    print("[INFO] adapter:", ADAPTER_DIR)
    print("[INFO] train:", TRAIN_PATH)
    print("[INFO] valid:", VALID_PATH)
    print("[INFO] output:", OUT_DIR)

    train_rows = load_jsonl(TRAIN_PATH)
    valid_rows = load_jsonl(VALID_PATH)

    print("[INFO] train rows:", len(train_rows))
    print("[INFO] valid rows:", len(valid_rows))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    base_model = prepare_model_for_kbit_training(base_model)

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
        is_trainable=True,
    )

    model.print_trainable_parameters()

    train_dataset = AnswerOnlyDataset(train_rows, tokenizer, MAX_LENGTH)
    valid_dataset = AnswerOnlyDataset(valid_rows, tokenizer, MAX_LENGTH)

    args = TrainingArguments(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=8e-5,
        warmup_steps=2,
        logging_steps=2,
        save_steps=10,
        save_total_limit=2,
        fp16=True,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
    )

    print("[TRAIN] start")
    trainer.train()

    print("[SAVE] adapter")
    trainer.model.save_pretrained(OUT_DIR / "adapter")
    tokenizer.save_pretrained(OUT_DIR / "adapter")

    print("[EVAL] final eval")
    metrics = trainer.evaluate()
    print(metrics)

    with (OUT_DIR / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("[OK] done")


if __name__ == "__main__":
    main()
