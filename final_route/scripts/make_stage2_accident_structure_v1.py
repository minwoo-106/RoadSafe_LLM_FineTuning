import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "data" / "processed" / "roadsafe_stage2_v4_gold_mix.jsonl"
OUT_DIR = ROOT / "final_route" / "data" / "stage2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "roadsafe_stage2_accident_structure_v1.jsonl"
OUT_REPORT = OUT_DIR / "roadsafe_stage2_accident_structure_v1_report.json"

REQUIRED_SECTIONS = [
    "[사고 유형]",
    "[핵심 판단 요소]",
    "[참고 과실 쟁점]",
    "[추가 확인 자료]",
    "[대처 방법]",
    "[주의]",
]

def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.splitlines())
    while "\n\n\n" in s:
        s = s.replace("\n\n\n", "\n\n")
    return s.strip()

def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    rows = []
    bad_rows = []

    for idx, line in enumerate(SRC.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue

        obj = json.loads(line)

        instruction = normalize_text(obj.get("instruction", ""))
        user_input = normalize_text(obj.get("input", ""))
        output = normalize_text(obj.get("output", ""))

        missing = [sec for sec in REQUIRED_SECTIONS if sec not in output]

        row = {
            "id": f"stage2_accident_structure_v1_{len(rows)+1:05d}",
            "stage": "stage2_accident_structure",
            "task_type": "sft",
            "instruction": instruction,
            "input": user_input,
            "output": output,
            "text": (
                "### 지시문\n"
                f"{instruction}\n\n"
                "### 사고 상황\n"
                f"{user_input}\n\n"
                "### 분석 답변\n"
                f"{output}"
            ),
            "missing_sections": missing,
        }

        if missing:
            bad_rows.append({
                "line": idx,
                "id": row["id"],
                "missing_sections": missing,
            })

        rows.append(row)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(SRC),
        "output": str(OUT_JSONL),
        "rows": len(rows),
        "bad_rows": len(bad_rows),
        "bad_row_details": bad_rows[:20],
        "required_sections": REQUIRED_SECTIONS,
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {OUT_JSONL}")
    print(f"[OK] wrote {OUT_REPORT}")
    print(f"[ROWS] {len(rows)}")
    print(f"[BAD_ROWS] {len(bad_rows)}")

    if bad_rows:
        print("[WARN] some rows missing required sections")
        for x in bad_rows[:10]:
            print(x)

if __name__ == "__main__":
    main()
