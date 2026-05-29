import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = ROOT / "final_route" / "outputs" / "stage3_user_instruction" / "full_v1" / "adapter"

instruction = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 쟁점, 대처 방법을 구조화하여 답변하라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라."
)

cases = [
    "자전거를 탄 채 일반 횡단보도를 건너다가 우회전하던 자동차와 부딪혔다. 자전거횡단도 표시는 없었다.",
    "앞차가 고속도로 1차로에서 갑자기 급제동해서 뒤에서 따라가던 내 차가 추돌했다.",
    "새벽에 보행자가 무단횡단을 하다가 정상 주행하던 차량과 충돌했다.",
    "빙판길에서 차량이 미끄러져 앞차를 들이받았다.",
    "사고났는데 누가 잘못이야?",
    "상대가 갑자기 끼어들었으니까 상대방 100% 맞지?",
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
        "### 사용자 질문\n"
        f"{case}\n\n"
        "### 답변\n"
    )

    print("=" * 100)
    print("[CASE]", case)
    print("-" * 100)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=520,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # 답변부만 잘라서 출력
    if "### 답변" in decoded:
        answer = decoded.split("### 답변", 1)[1].strip()
    else:
        answer = decoded.strip()

    # <END> 이후는 잘라냄
    if "<END>" in answer:
        answer = answer.split("<END>", 1)[0].strip() + "\n<END>"

    print(answer)
    print()
