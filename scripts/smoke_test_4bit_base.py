import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "meta-llama/Llama-3.2-3B"

print("[INFO] torch:", torch.__version__)
print("[INFO] cuda available:", torch.cuda.is_available())
print("[INFO] cuda:", torch.version.cuda)
print("[INFO] model:", MODEL_ID)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("[LOAD] tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[LOAD] model 4bit")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

print("[OK] model loaded")
print("[INFO] device:", next(model.parameters()).device)

prompt = """다음 도로교통 상황을 분석하고 관련 규칙, 핵심 쟁점, 대처 방법을 설명하라.

상황: 고속도로 1차로에서 정속 주행 중 뒤차가 경적을 울리다가 앞으로 추월한 뒤 급제동해서 사고가 났다.

답변:
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n===== GENERATED =====")
print(text[-1000:])
print("=====================")

print("[OK] smoke test finished")
