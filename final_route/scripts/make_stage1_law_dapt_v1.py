import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_TEXT_DIR = ROOT / "data" / "processed" / "raw_texts"

OUT_DIR = ROOT / "final_route" / "data" / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "roadsafe_stage1_law_dapt_v1.jsonl"
OUT_TXT = OUT_DIR / "roadsafe_stage1_law_dapt_v1.txt"
OUT_REPORT = OUT_DIR / "roadsafe_stage1_law_dapt_v1_report.json"

# Stage1은 법률 언어/기본 개념 적응 목적이므로
# 과실비율 사례보다는 도로교통법, 생활법령, 공공기관 해설 중심으로 사용한다.
INCLUDE_KEYWORDS = [
    "도로교통법",
    "교통_운전",
    "자전거_운전자",
    "긴급차량",
]

EXCLUDE_KEYWORDS = [
    "PM대자동차사고과실비율",
    "회전교차로사고_과실비율",
    "자동차사고_과실비율_인정기준",
]

MAX_CHARS = 2600
MIN_CHARS = 300


def is_target_file(path: Path) -> bool:
    name = path.name
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    return any(k in name for k in INCLUDE_KEYWORDS)


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = text.replace("\xa0", " ")
    text = text.replace("\t", " ")

    # 너무 많은 공백 정리
    text = re.sub(r"[ ]{2,}", " ", text)

    # 줄 단위 정리
    lines = []
    for line in text.splitlines():
        line = line.strip()

        if not line:
            lines.append("")
            continue

        # PDF 추출 잡음성 짧은 페이지 표시 제거
        if re.fullmatch(r"-?\s*\d+\s*-?", line):
            continue
        if re.fullmatch(r"페이지\s*\d+", line):
            continue

        lines.append(line)

    text = "\n".join(lines)

    # 빈 줄 과다 정리
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_paragraphs(text: str):
    # 빈 줄 기준 1차 분할
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 너무 짧은 문단은 뒤에 붙일 수 있도록 반환
    return parts


def pack_chunks(paragraphs):
    chunks = []
    buf = ""

    for p in paragraphs:
        if len(p) > MAX_CHARS:
            # 너무 긴 문단은 문장 단위로 쪼개기
            sentences = re.split(r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=니다\.)\s+|(?<=한다\.)\s+|(?<=된다\.)\s+", p)
            for s in sentences:
                s = s.strip()
                if not s:
                    continue

                if len(buf) + len(s) + 2 <= MAX_CHARS:
                    buf = (buf + "\n" + s).strip()
                else:
                    if len(buf) >= MIN_CHARS:
                        chunks.append(buf.strip())
                    buf = s
            continue

        if len(buf) + len(p) + 2 <= MAX_CHARS:
            buf = (buf + "\n\n" + p).strip()
        else:
            if len(buf) >= MIN_CHARS:
                chunks.append(buf.strip())
            buf = p

    if len(buf) >= MIN_CHARS:
        chunks.append(buf.strip())

    return chunks


def main():
    if not RAW_TEXT_DIR.exists():
        raise FileNotFoundError(f"raw text dir not found: {RAW_TEXT_DIR}")

    source_files = sorted([p for p in RAW_TEXT_DIR.glob("*.txt") if is_target_file(p)])

    if not source_files:
        raise RuntimeError("No target source files found for Stage1 Law DAPT.")

    rows = []
    txt_blocks = []
    source_stats = []

    for src in source_files:
        raw = src.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        paragraphs = split_into_paragraphs(cleaned)
        chunks = pack_chunks(paragraphs)

        source_stats.append({
            "source_file": src.name,
            "raw_chars": len(raw),
            "clean_chars": len(cleaned),
            "chunks": len(chunks),
        })

        for idx, chunk in enumerate(chunks, start=1):
            row = {
                "id": f"stage1_law_dapt_{len(rows)+1:05d}",
                "stage": "stage1_law_dapt",
                "task_type": "clm",
                "source_file": src.name,
                "chunk_index": idx,
                "text": chunk,
            }
            rows.append(row)
            txt_blocks.append(chunk)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    OUT_TXT.write_text("\n\n" + ("=" * 80) + "\n\n".join(txt_blocks), encoding="utf-8")

    lengths = [len(r["text"]) for r in rows]
    report = {
        "output_jsonl": str(OUT_JSONL),
        "output_txt": str(OUT_TXT),
        "rows": len(rows),
        "total_text_chars": sum(lengths),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "avg_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "source_files": source_stats,
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {OUT_JSONL}")
    print(f"[OK] wrote {OUT_TXT}")
    print(f"[OK] wrote {OUT_REPORT}")
    print(f"[ROWS] {report['rows']}")
    print(f"[TOTAL_CHARS] {report['total_text_chars']}")
    print(f"[AVG_CHARS] {report['avg_chars']}")
    print("[SOURCES]")
    for s in source_stats:
        print(f"- {s['source_file']} | chunks={s['chunks']} | clean_chars={s['clean_chars']}")


if __name__ == "__main__":
    main()
