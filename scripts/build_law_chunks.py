import json
import re
from pathlib import Path

IN_DIR = Path("data/processed/raw_texts")
OUT_PATH = Path("data/processed/roadsafe_law_chunks_v1.jsonl")

# 지금은 standards는 제외. 긴급차량 우선신호는 RoadSafe 사고상담 핵심에서 약간 멀어서 나중에 확장.
EXCLUDE_KEYWORDS = [
    "긴급차량_우선신호시스템",
]

SOURCE_HINTS = [
    ("도로교통법", "law"),
    ("교통_운전", "easylaw_traffic"),
    ("자전거_운전자", "easylaw_bicycle"),
    ("PM대자동차", "fault_pm"),
    ("회전교차로", "fault_roundabout"),
]

def category_for(name: str) -> str:
    for key, cat in SOURCE_HINTS:
        if key in name:
            return cat
    if "과실비율" in name:
        return "fault_ratio"
    return "unknown"

def clean_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"\r\n|\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def make_chunks(text: str, chunk_size=900, overlap=120):
    text = clean_text(text)
    chunks = []
    start = 0

    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()

        # 너무 짧거나 깨진 조각은 제외
        if len(chunk) >= 120:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks

rows = []
chunk_id = 0

for path in sorted(IN_DIR.glob("*.txt")):
    name = path.name

    if any(x in name for x in EXCLUDE_KEYWORDS):
        print(f"[SKIP] {name}")
        continue

    text = path.read_text(encoding="utf-8", errors="ignore")

    # 추출 실패 파일 제외
    if len(text.strip()) < 300:
        print(f"[SKIP_SMALL] {name} ({len(text.strip())} chars)")
        continue

    category = category_for(name)
    chunks = make_chunks(text)

    print(f"[SOURCE] {name} | category={category} | chunks={len(chunks)}")

    for i, chunk in enumerate(chunks):
        rows.append({
            "chunk_id": f"chunk_{chunk_id:05d}",
            "source_file": name,
            "category": category,
            "chunk_index": i,
            "text": chunk,
        })
        chunk_id += 1

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUT_PATH.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[OK] wrote {OUT_PATH}")
print(f"[OK] chunks: {len(rows)}")
