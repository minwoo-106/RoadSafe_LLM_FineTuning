import json
from pathlib import Path

gold_path = Path("data/processed/roadsafe_stage2_v4_gold.jsonl")
base_path = Path("data/processed/roadsafe_stage2_fault_scenario_v3_clean.jsonl")
out_path = Path("data/processed/roadsafe_stage2_v4_gold_mix.jsonl")

rows = []

for path in [gold_path, base_path]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

unique = []
seen = set()

for r in rows:
    if r["input"] not in seen:
        unique.append(r)
        seen.add(r["input"])

out_path.parent.mkdir(parents=True, exist_ok=True)

with out_path.open("w", encoding="utf-8") as f:
    for r in unique:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"[OK] wrote {out_path}")
print(f"[OK] rows: {len(unique)}")
print(f"[OK] gold first, duplicates removed")
