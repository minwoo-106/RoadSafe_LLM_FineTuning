import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "final_route/data/stage4/roadsafe_stage4_hardcase_v1.jsonl"
OUT_DIR = ROOT / "final_route/data/stage4/split_v1"
REPORT = OUT_DIR / "split_report.json"

SEED = 42

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

    # 32개 기준: train 28 / val 4
    train = rows[:28]
    val = rows[28:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "val.jsonl", val)

    report = {
        "source": str(SRC),
        "out_dir": str(OUT_DIR),
        "seed": SEED,
        "total": len(rows),
        "train": len(train),
        "val": len(val),
        "note": "Stage4 hardcase split. Golden eval remains separate.",
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Stage4 hardcase split done]")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
