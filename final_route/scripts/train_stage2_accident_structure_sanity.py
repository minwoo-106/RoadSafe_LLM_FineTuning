import os
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel, prepare_model_for_kbit_training


ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "meta-llama/Llama-3.2-3B"

STAGE1_ADAPTER_DIR = ROOT / "final_route" / "outputs" / "stage1_law_dapt" / "full_v1" / "adapter"

DATA_DIR = ROOT / "final_route" / "data" / "stage2" / "split_v1"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"

OUT_DIR = ROOT / "final_route" / "outputs" / "stage2_accident_structure" / "sanity_v1"
LOG_DIR = ROOT / "final_route" / "tensorboard" / "stage2_accident_structure_sanity_v1"

MAX_SEQ_LENGTH = 1024


def build_training_args():
    kwargs = dict(
        output_dir=str(OUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=10,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        logging_dir=str(LOG_DIR),
        logging_steps=1,
        eval_steps=5,
        save_steps=10,
        save_total_limit=1,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
        kwargs["logging_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"
        kwargs["save_strategy"] = "steps"
        kwargs["logging_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def main():
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    print("[INFO] ROOT:", ROOT)
    print("[INFO] STAGE1_ADAPTER_DIR:", STAGE1_ADAPTER_DIR)
    print("[INFO] TRAIN_FILE:", TRAIN_FILE)
    print("[INFO] VALID_FILE:", VALID_FILE)
    print("[INFO] OUT_DIR:", OUT_DIR)

    if not STAGE1_ADAPTER_DIR.exists():
        raise FileNotFoundError(STAGE1_ADAPTER_DIR)
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(TRAIN_FILE)
    if not VALID_FILE.exists():
        raise FileNotFoundError(VALID_FILE)

    print("[INFO] CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(TRAIN_FILE),
            "validation": str(VALID_FILE),
        },
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base)

    print("[INFO] loading Stage1 adapter as trainable...")
    model = PeftModel.from_pretrained(
        base,
        STAGE1_ADAPTER_DIR,
        is_trainable=True,
    )
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args = build_training_args()

    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_sig.parameters:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    print("[INFO] start Stage2 Accident Structure sanity training")
    train_result = trainer.train()
    print("[INFO] train_result:", train_result)

    metrics = trainer.evaluate()
    print("[INFO] eval metrics:", metrics)

    trainer.save_model(str(OUT_DIR / "adapter"))
    tokenizer.save_pretrained(str(OUT_DIR / "tokenizer"))

    print("[OK] saved adapter:", OUT_DIR / "adapter")
    print("[OK] saved tokenizer:", OUT_DIR / "tokenizer")


if __name__ == "__main__":
    main()
