import json
import sys
from pathlib import Path
from collections import Counter

required_keys = ["instruction", "input", "output"]
required_sections = ["[사고 유형]", "[핵심 판단 요소]", "[참고 과실 쟁점]", "[추가 확인 자료]", "[대처 방법]", "[주의]"]

if len(sys.argv) < 2:
    print("usage: python scripts/check_roadsafe_stage2_jsonl.py <jsonl_path>")
    raise SystemExit(1)

path = Path(sys.argv[1])
rows = []
errors = []

with path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as e:
            errors.append(f"line {i}: JSON parse error: {e}")
            continue

        for key in required_keys:
            if key not in obj:
                errors.append(f"line {i}: missing key {key}")

        output = obj.get("output", "")
        missing = [s for s in required_sections if s not in output]
        if missing:
            errors.append(f"line {i}: missing sections {missing}")

        bad_phrases = ["무조건 100%", "확정입니다", "반드시 이깁니다", "상대 100%"]
        for phrase in bad_phrases:
            if phrase in output:
                errors.append(f"line {i}: risky phrase found: {phrase}")

        rows.append(obj)

inputs = [r.get("input", "") for r in rows]
dups = [x for x, c in Counter(inputs).items() if c > 1]

print(f"[FILE] {path}")
print(f"[ROWS] {len(rows)}")
print(f"[UNIQUE_INPUTS] {len(set(inputs))}")

if dups:
    print(f"[WARN] duplicate inputs: {len(dups)}")
    for x in dups[:10]:
        print(" -", x)

lengths = [len(r.get("output", "")) for r in rows]
if lengths:
    print(f"[OUTPUT_LEN] min={min(lengths)} max={max(lengths)} avg={sum(lengths)//len(lengths)}")

if errors:
    print("\n[ERRORS]")
    for e in errors[:50]:
        print("-", e)
    raise SystemExit(1)

print("[OK] Stage2 JSONL valid")
