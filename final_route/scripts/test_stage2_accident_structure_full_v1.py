import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = ROOT / "final_route" / "outputs" / "stage2_accident_structure" / "full_v1" / "adapter"

instruction = "다음 교통사고 상황을 사고 유형, 핵심 판단 요소, 참고 과실 쟁점, 추가 확인 자료, 대처 방법 중심으로 분석하라."

cases = [
    "자전거를 타고 횡단보도를 건너다가 우회전하던 자동차와 부딪혔다. 자전거횡단도 표시는 없었다.",
    "앞차가 고속도로 1차로에서 갑자기 급제동해서 뒤에서 따라가던 내 차가 추돌했다.",
    "새벽에 보행자가 무단횡단을 하다가 정상 주행하던 차량과 충돌했다.",
    "빙판길에서 차량이 미끄러져 앞차를 들이받았다.",
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

for case in cases:
    prompt = (
        "### 지시문\n"
        f"{instruction}\n\n"
        "### 사고 상황\n"
        f"{case}\n\n"
        "### 분석 답변\n"
    )

    print("=" * 100)
    print("[CASE]", case)
    print("-" * 100)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False,
            repetition_penalty=1.18,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(decoded)
    print()
