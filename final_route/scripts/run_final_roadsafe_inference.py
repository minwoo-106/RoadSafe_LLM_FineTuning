import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from final_response_utils import finalize_answer, safety_check


ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "meta-llama/Llama-3.2-3B"

# Final candidate: Stage3 full_v2
ADAPTER_DIR = ROOT / "final_route" / "outputs" / "stage3_user_instruction" / "full_v2" / "adapter"

MAX_NEW_TOKENS = 700

INSTRUCTION = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 쟁점, 대처 방법을 구조화하여 답변하라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라. "
    "반드시 자연스러운 한국어로만 답변하고 영어, 일본어, 중국어, 깨진 문자를 섞지 말라. "
    "답변 마지막에는 <END>를 출력하라."
)


def make_prompt(user_input: str) -> str:
    return (
        "### 지시문\n"
        f"{INSTRUCTION}\n\n"
        "### 사용자 질문\n"
        f"{user_input}\n\n"
        "### 답변\n"
    )


def postprocess_raw(decoded: str) -> str:
    if "### 답변" in decoded:
        answer = decoded.split("### 답변", 1)[1].strip()
    else:
        answer = decoded.strip()

    if "<END>" in answer:
        answer = answer.split("<END>", 1)[0].strip() + "\n<END>"

    return answer.strip()


def load_model():
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(f"Adapter not found: {ADAPTER_DIR}")

    print("[INFO] loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] loading base model in 4bit...")
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

    print("[INFO] loading Stage3 full_v2 adapter...")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    return tokenizer, model


def generate_answer(tokenizer, model, user_input: str):
    prompt = make_prompt(user_input)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    raw_answer = postprocess_raw(decoded)

    final_answer, check = finalize_answer(raw_answer, user_input=user_input)

    return {
        "input": user_input,
        "raw_answer": raw_answer,
        "final_answer": final_answer,
        "safety_check": check,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, default=None)
    parser.add_argument("--show-raw", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model()

    if args.question:
        questions = [args.question]
    else:
        questions = [
            "자전거를 탄 채 일반 횡단보도를 건너다가 우회전하던 자동차와 부딪혔다. 자전거횡단도 표시는 없었다.",
            "상대가 갑자기 끼어들었으니까 상대방 100% 맞지?",
            "고속도로에서 사고 후 차가 도로 한가운데 멈췄는데 2차 사고가 날까 봐 무섭다.",
            "사고났는데 누가 잘못이야?",
            "브레이크가 갑자기 안 들어서 앞차를 들이받았어. 고장이면 괜찮은 거야?",
        ]

    for idx, q in enumerate(questions, 1):
        print("\n" + "=" * 100)
        print(f"[QUESTION {idx}]")
        print(q)

        result = generate_answer(tokenizer, model, q)

        if args.show_raw:
            print("\n[RAW ANSWER]")
            print(result["raw_answer"])

        print("\n[FINAL ANSWER]")
        print(result["final_answer"])

        print("\n[SAFETY CHECK]")
        print(json.dumps(result["safety_check"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
