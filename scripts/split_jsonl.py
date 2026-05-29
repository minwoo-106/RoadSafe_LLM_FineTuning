import json
import random
import sys
from pathlib import Path

if len(sys.argv) < 3:
    print("usage: python scripts/split_jsonl.py <input_jsonl> <output_dir>")
    raise SystemExit(1)

input_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)

rows = []
with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

random.seed(42)
random.shuffle(rows)

n = len(rows)
test_n = max(1, round(n * 0.1))
valid_n = max(1, round(n * 0.1))
train_n = n - valid_n - test_n

splits = {
    "train": rows[:train_n],
    "valid": rows[train_n:train_n + valid_n],
    "test": rows[train_n + valid_n:],
}

for name, data in splits.items():
    out_path = out_dir / f"{name}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] {name}: {len(data)} -> {out_path}")

print(f"[OK] total: {n}")
