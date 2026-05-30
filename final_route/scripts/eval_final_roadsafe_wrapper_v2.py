import json
from pathlib import Path
from collections import Counter

from run_final_roadsafe_inference import load_model, generate_answer


ROOT = Path(__file__).resolve().parents[2]

EVAL_FILE = ROOT / "final_route/data/eval/stage3_golden_eval_v1.jsonl"
OUT_DIR = ROOT / "final_route/reports/final_eval"
OUT_JSONL = OUT_DIR / "final_wrapper_eval_v2_results.jsonl"
OUT_REPORT = OUT_DIR / "final_wrapper_eval_v2_report.json"


REQUIRED_SECTIONS = [
    "[상황 요약]",
    "[사고 유형]",
    "[사실관계 정리]",
    "[확인 필요]",
    "[관련 근거]",
    "[핵심 쟁점]",
    "[대처 방법]",
    "[주의]",
]


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def get_keywords(row):
    for key in ["keywords", "expected_keywords", "target_keywords"]:
        if key in row and isinstance(row[key], list):
            return row[key]
    return []


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(EVAL_FILE)
    tokenizer, model = load_model()

    results = []

    for idx, row in enumerate(rows, 1):
        user_input = row.get("input", row.get("question", "")).strip()
        category = row.get("category", "unknown")
        item_id = row.get("id", f"final_eval_{idx:03d}")

        print(f"[{idx}/{len(rows)}] {item_id} | {category}")

        result = generate_answer(tokenizer, model, user_input)
        final_answer = result["final_answer"]
        check = result["safety_check"]

        keywords = get_keywords(row)
        keyword_found = [kw for kw in keywords if kw in final_answer]

        section_found = [sec for sec in REQUIRED_SECTIONS if sec in final_answer]
        section_missing = [sec for sec in REQUIRED_SECTIONS if sec not in final_answer]

        results.append({
            "id": item_id,
            "category": category,
            "input": user_input,
            "pass": check.get("pass"),
            "fallback_used": check.get("fallback_used"),
            "forbidden": check.get("forbidden"),
            "bad_korean": check.get("bad_korean"),
            "missing_sections": check.get("missing_sections"),
            "has_end": check.get("has_end"),
            "section_found_count": len(section_found),
            "section_missing": section_missing,
            "keyword_total": len(keywords),
            "keyword_found_count": len(keyword_found),
            "keyword_found": keyword_found,
            "raw_answer": result["raw_answer"],
            "final_answer": final_answer,
        })

    OUT_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8"
    )

    total = len(results)
    category_counts = Counter(r["category"] for r in results)

    by_category = {}
    for category in sorted(category_counts):
        subset = [r for r in results if r["category"] == category]
        by_category[category] = {
            "count": len(subset),
            "pass": sum(1 for r in subset if r["pass"]),
            "fallback_used": sum(1 for r in subset if r["fallback_used"]),
            "has_end": sum(1 for r in subset if r["has_end"]),
            "section_full": sum(1 for r in subset if not r["section_missing"]),
            "bad_korean_clean": sum(1 for r in subset if not r["bad_korean"]),
            "forbidden_clean": sum(1 for r in subset if not r["forbidden"]),
        }

    keyword_rows = [r for r in results if r["keyword_total"] > 0]
    avg_keyword_recall = (
        sum(r["keyword_found_count"] / r["keyword_total"] for r in keyword_rows) / len(keyword_rows)
        if keyword_rows else None
    )

    report = {
        "total": total,
        "pass": sum(1 for r in results if r["pass"]),
        "fallback_used": sum(1 for r in results if r["fallback_used"]),
        "fallback_rate": round(sum(1 for r in results if r["fallback_used"]) / total, 4) if total else 0,
        "has_end": sum(1 for r in results if r["has_end"]),
        "section_full": sum(1 for r in results if not r["section_missing"]),
        "bad_korean_clean": sum(1 for r in results if not r["bad_korean"]),
        "forbidden_clean": sum(1 for r in results if not r["forbidden"]),
        "avg_section_found": round(sum(r["section_found_count"] for r in results) / total, 3) if total else 0,
        "avg_keyword_recall": round(avg_keyword_recall, 3) if avg_keyword_recall is not None else None,
        "by_category": by_category,
        "output_jsonl": str(OUT_JSONL),
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[FINAL WRAPPER EVAL V2 REPORT]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
