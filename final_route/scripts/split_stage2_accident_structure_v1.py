import json
import random
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "final_route" / "data" / "stage2" / "roadsafe_stage2_accident_structure_v1.jsonl"
out_dir = ROOT / "final_route" / "data" / "stage2" / "split_v1"
out_dir.mkdir(parents=True, exist_ok=True)

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
random.shuffle(rows)

n = len(rows)
n_train = int(n * 0.8)
n_valid = int(n * 0.1)

splits = {
    "train": rows[:n_train],
    "valid": rows[n_train:n_train + n_valid],
    "test": rows[n_train + n_valid:],
}

for name, data in splits.items():
    path = out_dir / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[OK] {name}: {len(data)} -> {path}")

print("[OK] total:", sum(len(v) for v in splits.values()))
