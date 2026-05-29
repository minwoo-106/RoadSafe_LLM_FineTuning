import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


MODEL_ID = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = Path("outputs/roadsafe_stage2_v4_gold_mix_from_stage1_qlora/adapter")
PROMPT_PATH = Path("data/processed/roadsafe_stage2_eval_prompts_v1.jsonl")

OUT_DIR = Path("outputs/eval_stage2_v4_gold_mix_strict")
OUT_JSONL = OUT_DIR / "roadsafe_stage2_v4_gold_mix_strict_eval_v1.jsonl"
OUT_TXT = OUT_DIR / "roadsafe_stage2_v4_gold_mix_strict_eval_v1.txt"

OUT_DIR.mkdir(parents=True, exist_ok=True)

print("[INFO] base model:", MODEL_ID)
print("[INFO] adapter:", ADAPTER_DIR)
print("[INFO] prompt file:", PROMPT_PATH)
print("[INFO] torch:", torch.__version__)
print("[INFO] cuda available:", torch.cuda.is_available())

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("[LOAD] tokenizer")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("[LOAD] base model 4bit")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    dtype=torch.float16,
)

print("[LOAD] adapter")
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model.eval()

rows = []
with PROMPT_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

results = []

for i, row in enumerate(rows, 1):
    eval_id = row["id"]
    user_prompt = row["prompt"]

    prompt = f"""다음 교통사고 상황을 사고 유형, 핵심 판단 요소, 참고 과실 쟁점, 추가 확인 자료, 대처 방법 중심으로 분석하라.

작성 규칙:
- 입력 상황에 없는 사실을 만들지 말 것.
- 자동차, 자전거, 보행자, PM의 역할을 바꾸지 말 것.
- 과실비율을 특정 숫자로 단정하지 말 것.
- "무조건", "확정", "반드시 이김" 같은 단정 표현을 피할 것.
- 문장이 어색하면 짧고 명확한 표현으로 쓸 것.
- 자전거도로 사고에서는 자동차가 자전거도로를 주행했다고 단정하지 말고, 차량의 우회전 경로가 자전거 진행 경로와 교차했는지 확인할 것.
- PM 횡단보도 사고에서는 탑승 상태인지, 내려서 끌었는지, 자전거횡단도 유무를 확인할 것.
- 신호 없는 교차로 사고에서는 선진입, 대로·소로, 우측도로, 직진·좌회전 관계를 확인할 것.

상황: {user_prompt}

답변:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=520,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    item = {
        "id": eval_id,
        "prompt": user_prompt,
        "model": "roadsafe_stage2_v4_gold_mix_from_stage1_qlora",
        "answer": answer,
    }
    results.append(item)

    print(f"\n===== {i}/{len(rows)} {eval_id} =====")
    print(user_prompt)
    print("--- answer ---")
    print(answer[:1500])

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with OUT_TXT.open("w", encoding="utf-8") as f:
    for item in results:
        f.write(f"===== {item['id']} =====\n")
        f.write(f"[PROMPT]\n{item['prompt']}\n\n")
        f.write(f"[ANSWER]\n{item['answer']}\n\n")

print("\n[OK] saved:", OUT_JSONL)
print("[OK] saved:", OUT_TXT)
