import json
import sys
from pathlib import Path
from collections import Counter

required_keys = ["instruction", "input", "output"]
required_sections = ["[상황 요약]", "[핵심 쟁점]", "[관련 규칙]", "[대처 방법]", "[주의]"]

if len(sys.argv) < 2:
    print("usage: python scripts/check_roadsafe_jsonl.py <jsonl_path>")
    raise SystemExit(1)

path = Path(sys.argv[1])
if not path.exists():
    print(f"[ERROR] file not found: {path}")
    raise SystemExit(1)

rows = []
errors = []

with path.open("r", encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        try:
            obj = json.loads(line)
        except Exception as e:
            errors.append(f"line {i}: JSON parse error: {e}")
            continue

        for key in required_keys:
            if key not in obj:
                errors.append(f"line {i}: missing key {key}")

        if "output" in obj:
            missing_sections = [s for s in required_sections if s not in obj["output"]]
            if missing_sections:
                errors.append(f"line {i}: missing sections {missing_sections}")

        rows.append(obj)

inputs = [r.get("input", "") for r in rows]
duplicate_inputs = [x for x, c in Counter(inputs).items() if c > 1]

print(f"[FILE] {path}")
print(f"[ROWS] {len(rows)}")
print(f"[UNIQUE_INPUTS] {len(set(inputs))}")

if duplicate_inputs:
    print(f"[WARN] duplicate inputs: {len(duplicate_inputs)}")
    for x in duplicate_inputs[:10]:
        print(" -", x)

lengths = [len(r.get("output", "")) for r in rows]
if lengths:
    print(f"[OUTPUT_LEN] min={min(lengths)} max={max(lengths)} avg={sum(lengths)//len(lengths)}")

if errors:
    print("\n[ERRORS]")
    for e in errors[:30]:
        print("-", e)
    raise SystemExit(1)

print("[OK] JSONL valid")
