import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = ROOT / "final_route" / "outputs" / "stage3_user_instruction" / "full_v3" / "adapter"

EVAL_FILE = ROOT / "final_route" / "data" / "eval" / "stage3_golden_eval_v1.jsonl"
OUT_DIR = ROOT / "final_route" / "reports" / "stage3_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "stage3_golden_eval_v3_promptfix_results.jsonl"
OUT_REPORT = OUT_DIR / "stage3_golden_eval_v3_promptfix_report.json"

INSTRUCTION = (
    "사용자의 교통사고 상담 질문을 읽고 사고 상황을 구조화하여 답변하라. "
    "반드시 [상황 요약], [사고 유형], [사실관계 정리], [확인 필요], [관련 근거], "
    "[핵심 쟁점], [대처 방법], [주의] 섹션을 이 순서대로 포함하라. "
    "각 섹션은 짧고 명확하게 작성하고, 답변을 과도하게 늘리지 말라. "
    "반드시 자연스러운 한국어로만 답변하고 영어 표현, 깨진 문자, 외국어 문자를 섞지 말라. "
    "과실비율, 법적 책임, 처벌 가능성, 보험 결과를 단정하지 말라. "
    "정보가 부족하면 추가 확인 자료가 필요하다고 안내하라. "
    "답변 마지막에는 반드시 <END>를 출력하라."
)

BAD_KOREAN_PATTERNS = [
    "率", "ス", "が", "책им", "사ogo",
    "횡담", "횽단", "횃단", "횡断",
    "자전기", "자전구", "쌍점", "숾절",
    "블랙박ス", "速度", "책им지", "교통사ogo",
]

def contains_any(text, items):
    return [x for x in items if x in text]

def section_score(answer, expected_sections):
    missing = [sec for sec in expected_sections if sec not in answer]
    return len(expected_sections) - len(missing), missing

def keyword_score(answer, expected_keywords):
    found = [kw for kw in expected_keywords if kw in answer]
    missing = [kw for kw in expected_keywords if kw not in answer]
    return len(found), found, missing

def repetition_score(answer):
    # 같은 문장 반복 간단 탐지
    sentences = re.split(r"(?<=[.!?。])\s+|\n+", answer)
    cleaned = [s.strip() for s in sentences if len(s.strip()) >= 8]
    seen = set()
    repeated = []
    for s in cleaned:
        if s in seen and s not in repeated:
            repeated.append(s)
        seen.add(s)
    return repeated

def make_prompt(user_input):
    return (
        "### 지시문\n"
        f"{INSTRUCTION}\n\n"
        "### 사용자 질문\n"
        f"{user_input}\n\n"
        "### 답변\n"
    )

def postprocess(decoded):
    if "### 답변" in decoded:
        answer = decoded.split("### 답변", 1)[1].strip()
    else:
        answer = decoded.strip()

    if "<END>" in answer:
        answer = answer.split("<END>", 1)[0].strip() + "\n<END>"

    return answer.strip()

def main():
    rows = [json.loads(line) for line in EVAL_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]

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

    results = []

    for i, row in enumerate(rows, start=1):
        prompt = make_prompt(row["input"])
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=900,
                do_sample=False,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = postprocess(decoded)

        sec_found, sec_missing = section_score(answer, row["expected_sections"])
        kw_found_count, kw_found, kw_missing = keyword_score(answer, row["expected_keywords"])
        forbidden_found = contains_any(answer, row["forbidden_phrases"])
        bad_korean_found = contains_any(answer, BAD_KOREAN_PATTERNS)
        repeated = repetition_score(answer)
        has_end = "<END>" in answer

        result = {
            "id": row["id"],
            "category": row["category"],
            "input": row["input"],
            "answer": answer,
            "section_total": len(row["expected_sections"]),
            "section_found": sec_found,
            "section_missing": sec_missing,
            "keyword_total": len(row["expected_keywords"]),
            "keyword_found_count": kw_found_count,
            "keyword_found": kw_found,
            "keyword_missing": kw_missing,
            "forbidden_found": forbidden_found,
            "bad_korean_found": bad_korean_found,
            "repeated_sentences": repeated,
            "has_end_token": has_end,
            "pass_section": len(sec_missing) == 0,
            "pass_forbidden": len(forbidden_found) == 0,
            "pass_end": has_end,
            "pass_bad_korean": len(bad_korean_found) == 0,
        }

        results.append(result)

        print(f"[{i:02d}/{len(rows)}] {row['id']} | sections {sec_found}/{len(row['expected_sections'])} | keywords {kw_found_count}/{len(row['expected_keywords'])} | END {has_end} | forbidden {len(forbidden_found)} | badko {len(bad_korean_found)}")

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    report = {
        "total": total,
        "section_pass": sum(r["pass_section"] for r in results),
        "end_pass": sum(r["pass_end"] for r in results),
        "forbidden_pass": sum(r["pass_forbidden"] for r in results),
        "bad_korean_pass": sum(r["pass_bad_korean"] for r in results),
        "avg_section_found": round(sum(r["section_found"] for r in results) / total, 3),
        "avg_keyword_found": round(sum(r["keyword_found_count"] for r in results) / total, 3),
        "avg_keyword_recall": round(
            sum(r["keyword_found_count"] / max(r["keyword_total"], 1) for r in results) / total,
            3
        ),
        "by_category": {},
        "output_jsonl": str(OUT_JSONL),
    }

    for r in results:
        group = r["category"].split("_")[0]
        report["by_category"].setdefault(group, {"count": 0, "section_pass": 0, "end_pass": 0, "forbidden_pass": 0, "bad_korean_pass": 0})
        report["by_category"][group]["count"] += 1
        report["by_category"][group]["section_pass"] += int(r["pass_section"])
        report["by_category"][group]["end_pass"] += int(r["pass_end"])
        report["by_category"][group]["forbidden_pass"] += int(r["pass_forbidden"])
        report["by_category"][group]["bad_korean_pass"] += int(r["pass_bad_korean"])

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[REPORT]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
