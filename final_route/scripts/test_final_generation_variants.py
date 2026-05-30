import json
from pathlib import Path

import torch

from run_final_roadsafe_inference import load_model, postprocess_raw
from final_response_utils import finalize_answer, detect_bad_korean, detect_forbidden


INSTRUCTION_BASE = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 쟁점, 대처 방법을 구조화하여 답변하라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라."
)

INSTRUCTION_STRICT_KO = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 쟁점, 대처 방법을 구조화하여 답변하라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라. "
    "반드시 자연스러운 한국어로만 답변하고 영어, 일본어, 중국어, 깨진 문자를 섞지 말라. "
    "답변 마지막에는 <END>를 출력하라."
)

QUESTIONS = [
    "자전거를 탄 채 일반 횡단보도를 건너다가 우회전하던 자동차와 부딪혔다. 자전거횡단도 표시는 없었다.",
    "상대가 갑자기 끼어들었으니까 상대방 100% 맞지?",
    "고속도로에서 사고 후 차가 도로 한가운데 멈췄는데 2차 사고가 날까 봐 무섭다.",
    "사고났는데 누가 잘못이야?",
    "브레이크가 갑자기 안 들어서 앞차를 들이받았어. 고장이면 괜찮은 거야?",
]

VARIANTS = [
    {
        "name": "current_rep115_ngram4",
        "instruction": INSTRUCTION_BASE,
        "max_new_tokens": 900,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 4,
    },
    {
        "name": "plain_greedy_700",
        "instruction": INSTRUCTION_BASE,
        "max_new_tokens": 700,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    },
    {
        "name": "plain_greedy_550",
        "instruction": INSTRUCTION_BASE,
        "max_new_tokens": 550,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    },
    {
        "name": "low_rep_103_700",
        "instruction": INSTRUCTION_BASE,
        "max_new_tokens": 700,
        "repetition_penalty": 1.03,
        "no_repeat_ngram_size": 0,
    },
    {
        "name": "strict_ko_plain_700",
        "instruction": INSTRUCTION_STRICT_KO,
        "max_new_tokens": 700,
        "repetition_penalty": 1.0,
        "no_repeat_ngram_size": 0,
    },
    {
        "name": "strict_ko_lowrep_700",
        "instruction": INSTRUCTION_STRICT_KO,
        "max_new_tokens": 700,
        "repetition_penalty": 1.03,
        "no_repeat_ngram_size": 0,
    },
]


def make_prompt(instruction: str, user_input: str) -> str:
    return (
        f"### 지시문\n{instruction}\n\n"
        f"### 사용자 질문\n{user_input}\n\n"
        "### 답변\n"
    )


def generate_raw(tokenizer, model, question: str, variant: dict) -> str:
    prompt = make_prompt(variant["instruction"], question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    gen_kwargs = {
        "max_new_tokens": variant["max_new_tokens"],
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if variant["repetition_penalty"] != 1.0:
        gen_kwargs["repetition_penalty"] = variant["repetition_penalty"]

    if variant["no_repeat_ngram_size"] > 0:
        gen_kwargs["no_repeat_ngram_size"] = variant["no_repeat_ngram_size"]

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return postprocess_raw(decoded)


def main():
    tokenizer, model = load_model()

    all_results = []

    for variant in VARIANTS:
        print("\n" + "#" * 100)
        print("[VARIANT]", variant["name"])

        fallback_count = 0
        raw_bad_count = 0
        raw_forbidden_count = 0
        pass_count = 0

        for i, q in enumerate(QUESTIONS, 1):
            raw = generate_raw(tokenizer, model, q, variant)
            raw_bad = detect_bad_korean(raw)
            raw_forbidden = detect_forbidden(raw)

            final, check = finalize_answer(raw, user_input=q)

            fallback_used = check.get("fallback_used", False)
            fallback_count += int(bool(fallback_used))
            raw_bad_count += int(bool(raw_bad))
            raw_forbidden_count += int(bool(raw_forbidden))
            pass_count += int(bool(check.get("pass")))

            print("-" * 100)
            print(f"[{i}] {q}")
            print("raw_bad:", raw_bad)
            print("raw_forbidden:", raw_forbidden)
            print("final_pass:", check.get("pass"))
            print("fallback_used:", fallback_used)
            print("raw_preview:")
            print(raw[:500].replace("\n", " ") + ("..." if len(raw) > 500 else ""))

            all_results.append({
                "variant": variant["name"],
                "question": q,
                "raw_bad": raw_bad,
                "raw_forbidden": raw_forbidden,
                "final_pass": check.get("pass"),
                "fallback_used": fallback_used,
                "raw_answer": raw,
                "final_answer": final,
            })

        print("\n[SUMMARY]", variant["name"])
        print(json.dumps({
            "total": len(QUESTIONS),
            "final_pass": pass_count,
            "fallback_count": fallback_count,
            "raw_bad_count": raw_bad_count,
            "raw_forbidden_count": raw_forbidden_count,
        }, ensure_ascii=False, indent=2))

    out = Path("final_route/reports/final_eval/final_generation_variants_v1_results.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_results) + "\n",
        encoding="utf-8"
    )
    print("\n[OK] saved:", out)


if __name__ == "__main__":
    main()
