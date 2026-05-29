import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v3.jsonl"
OUT_DIR = ROOT / "final_route/data/stage3/split_v3"
REPORT = ROOT / "final_route/data/stage3/split_v3/split_report.json"

SEED = 42
TRAIN_RATIO = 0.85
VAL_RATIO = 0.10
# TEST_RATIO = 0.05

def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def write_jsonl(path: Path, rows):
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8"
    )

def main():
    rows = read_jsonl(SRC)

    random.seed(SEED)
    random.shuffle(rows)

    n = len(rows)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)

    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    report = {
        "source": str(SRC),
        "out_dir": str(OUT_DIR),
        "seed": SEED,
        "total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio_actual": round(len(test) / n, 4),
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Stage3 v3 split done]")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
