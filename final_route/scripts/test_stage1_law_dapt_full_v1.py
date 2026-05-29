import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = ROOT / "final_route" / "outputs" / "stage1_law_dapt" / "full_v1" / "adapter"

prompts = [
    "도로교통법에서 보행자의 도로 횡단과 관련하여",
    "자동차등의 운전자는 도로에서 안전하게 운전하기 위하여",
    "자전거등의 운전자가 횡단보도를 이용하여 도로를 횡단하는 경우",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.eval()

for prompt in prompts:
    print("=" * 100)
    print("[PROMPT]", prompt)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print()
