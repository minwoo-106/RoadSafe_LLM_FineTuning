import json
import subprocess
from pathlib import Path

PROMPT_PATH = Path("data/processed/roadsafe_rag_controlled_eval_v1.jsonl")
OUT_DIR = Path("outputs/eval_rag_controlled_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TXT = OUT_DIR / "rag_controlled_eval_v1.txt"
OUT_JSONL = OUT_DIR / "rag_controlled_eval_v1.jsonl"

required_by_type = {
    "pm_crosswalk": ["내려서 끌고", "자전거횡단도", "횡단보도"],
    "bicycle_crossing": ["자전거횡단도", "일시정지", "횡단"],
    "highway_lane": ["고속도로 1차로", "급제동", "난폭운전"],
    "right_turn_pedestrian": ["우회전", "횡단보도", "보행자"],
    "uncontrolled_intersection": ["선진입", "도로 폭", "우측도로"],
}

bad_phrases = [
    "무조건 100%",
    "상대 100%",
    "확정입니다",
    "반드시 이깁니다",
    "자전거도는",
    "자동차가 자전거도로를 우회전",
]

rows = [json.loads(line) for line in PROMPT_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
results = []

with OUT_TXT.open("w", encoding="utf-8") as txt:
    for row in rows:
        cmd = [
            "python",
            "scripts/answer_with_retrieval_v3_controlled.py",
            row["prompt"],
            "--top_k_rules", "3",
            "--top_k_chunks", "4",
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        output = proc.stdout

        answer_marker = "[CONTROLLED ANSWER]"
        answer = output.split(answer_marker, 1)[1].strip() if answer_marker in output else output.strip()

        required = required_by_type.get(row["type"], [])
        missing_required = [x for x in required if x not in answer]
        found_bad = [x for x in bad_phrases if x in answer]

        item = {
            "id": row["id"],
            "type": row["type"],
            "prompt": row["prompt"],
            "missing_required": missing_required,
            "found_bad": found_bad,
            "answer": answer,
        }
        results.append(item)

        txt.write(f"===== {row['id']} / {row['type']} =====\n")
        txt.write(f"[PROMPT]\n{row['prompt']}\n\n")
        txt.write(f"[MISSING_REQUIRED] {missing_required}\n")
        txt.write(f"[FOUND_BAD] {found_bad}\n\n")
        txt.write(f"[ANSWER]\n{answer}\n\n")

with OUT_JSONL.open("w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("[OK] saved", OUT_TXT)
print("[OK] saved", OUT_JSONL)

print("\n[SUMMARY]")
for item in results:
    if item["missing_required"] or item["found_bad"]:
        print(f"[WARN] {item['id']} type={item['type']} missing={item['missing_required']} bad={item['found_bad']}")

print("[OK] eval done")
