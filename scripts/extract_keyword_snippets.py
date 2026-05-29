from pathlib import Path
import re

TEXT_DIR = Path("data/processed/raw_texts")
OUT = Path("data/processed/keyword_snippets.txt")

keywords = [
    "난폭운전",
    "보복운전",
    "급제동",
    "안전거리",
    "앞지르기",
    "고속도로",
    "자전거횡단도",
    "자전거도로",
    "전조등",
    "사고 발생",
    "교통사고 발생",
    "횡단보도",
    "우회전",
]

lines = []

for path in sorted(TEXT_DIR.glob("*.txt")):
    text = path.read_text(encoding="utf-8", errors="ignore")
    compact = re.sub(r"\s+", " ", text)

    lines.append(f"\n\n===== {path.name} =====\n")

    for kw in keywords:
        found = False
        for m in re.finditer(re.escape(kw), compact):
            start = max(0, m.start() - 180)
            end = min(len(compact), m.end() + 260)
            snippet = compact[start:end]
            lines.append(f"\n--- keyword: {kw} ---\n{snippet}\n")
            found = True
            break

        if not found:
            pass

OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"[OK] wrote {OUT}")
