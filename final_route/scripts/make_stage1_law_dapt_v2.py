import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_TEXT_DIR = ROOT / "data" / "processed" / "raw_texts"

OUT_DIR = ROOT / "final_route" / "data" / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "roadsafe_stage1_law_dapt_v2.jsonl"
OUT_TXT = OUT_DIR / "roadsafe_stage1_law_dapt_v2.txt"
OUT_REPORT = OUT_DIR / "roadsafe_stage1_law_dapt_v2_report.json"

# Stage1은 법률 언어 적응이 목적.
# 과실비율/사례성 문서는 Stage2로 보내고,
# Stage1에서는 도로교통법 + 생활법령 중심으로 시작한다.
INCLUDE_KEYWORDS = [
    "도로교통법",
    "교통_운전",
    "자전거_운전자",
]

# 긴급차량 표준규격은 통신/인증/장비 규격 텍스트가 많아
# Law DAPT v2에서는 제외한다.
EXCLUDE_KEYWORDS = [
    "긴급차량",
    "PM대자동차사고과실비율",
    "회전교차로사고_과실비율",
    "자동차사고_과실비율_인정기준",
]

MAX_CHARS = 2200
MIN_CHARS = 350


def is_target_file(path: Path) -> bool:
    name = path.name
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    return any(k in name for k in INCLUDE_KEYWORDS)


def normalize_line(line: str) -> str:
    line = line.replace("\ufeff", " ")
    line = line.replace("\xa0", " ")
    line = line.replace("\u0000", " ")
    line = line.replace("\t", " ")

    # PDF 목차 점선 제거
    line = re.sub(r"·{2,}", " ", line)
    line = re.sub(r"\.{4,}", " ", line)

    # 특수 bullet 일부 정리
    line = line.replace("Ÿ", "- ")

    # 공백 정리
    line = re.sub(r"[ ]{2,}", " ", line)
    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return False

    # 페이지 마커
    if re.match(r"^--- PAGE \d+ ---$", line):
        return True

    # 숫자만 있는 줄
    if re.fullmatch(r"-?\s*\d+\s*-?", line):
        return True

    # 목차성 줄
    if line in {"목 차", "목차", "차 례", "차례"}:
        return True

    # 점선/기호 비중이 너무 높은 줄
    if len(line) >= 20:
        symbols = sum(1 for ch in line if ch in "·.-_—–•")
        if symbols / len(line) > 0.35:
            return True

    return False


def clean_text(text: str) -> str:
    lines = []
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if is_noise_line(line):
            continue
        lines.append(line)

    text = "\n".join(lines)

    # 페이지 마커 혹시 남은 것 제거
    text = re.sub(r"--- PAGE \d+ ---", "\n", text)

    # 조항/장/절 앞에 줄바꿈 보강
    text = re.sub(r"(?<!\n)(제\s*\d+\s*조)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(제\s*\d+\s*장)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(제\s*\d+\s*절)", r"\n\1", text)

    # 생활법령 질문형/소제목 분리 보강
    text = re.sub(r"(?<!\n)(Q\.|A\.)", r"\n\1", text)

    # 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def split_paragraphs(text: str):
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    # 너무 긴 문단은 줄 단위로 재분할
    refined = []
    for p in parts:
        if len(p) <= MAX_CHARS:
            refined.append(p)
        else:
            lines = [x.strip() for x in p.splitlines() if x.strip()]
            refined.extend(lines)

    return refined


def split_hard(text: str, max_chars: int):
    # 문장부호/종결어미 기준으로 쪼갠 뒤, 그래도 길면 강제 분할
    sentence_pattern = r"(?<=[.!?。])\s+|(?<=다\.)\s+|(?<=니다\.)\s+|(?<=한다\.)\s+|(?<=된다\.)\s+|(?<=함\.)\s+"
    pieces = []
    for part in re.split(sentence_pattern, text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            pieces.append(part)
        else:
            for i in range(0, len(part), max_chars):
                sub = part[i:i + max_chars].strip()
                if sub:
                    pieces.append(sub)
    return pieces


def pack_chunks(paragraphs):
    chunks = []
    buf = ""

    for p in paragraphs:
        p = p.strip()
        if not p:
            continue

        parts = split_hard(p, MAX_CHARS) if len(p) > MAX_CHARS else [p]

        for part in parts:
            if len(buf) + len(part) + 2 <= MAX_CHARS:
                buf = (buf + "\n\n" + part).strip()
            else:
                if len(buf) >= MIN_CHARS:
                    chunks.append(buf.strip())
                buf = part

            # 그래도 buffer가 너무 길면 바로 자름
            while len(buf) > MAX_CHARS:
                chunks.append(buf[:MAX_CHARS].strip())
                buf = buf[MAX_CHARS:].strip()

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
        paragraphs = split_paragraphs(cleaned)
        chunks = pack_chunks(paragraphs)

        # 최종 안전 검사: 너무 짧거나 목차성 노이즈 강한 chunk 제거
        filtered = []
        for ch in chunks:
            if len(ch) < MIN_CHARS:
                continue
            if "목 차" in ch[:100] or "목차" in ch[:100]:
                continue
            filtered.append(ch)

        source_stats.append({
            "source_file": src.name,
            "raw_chars": len(raw),
            "clean_chars": len(cleaned),
            "chunks": len(filtered),
        })

        for idx, chunk in enumerate(filtered, start=1):
            row = {
                "id": f"stage1_law_dapt_v2_{len(rows)+1:05d}",
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

    OUT_TXT.write_text(("\n\n" + "=" * 80 + "\n\n").join(txt_blocks), encoding="utf-8")

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
    print(f"[MIN_CHARS] {report['min_chars']}")
    print(f"[MAX_CHARS] {report['max_chars']}")
    print(f"[AVG_CHARS] {report['avg_chars']}")
    print("[SOURCES]")
    for s in source_stats:
        print(f"- {s['source_file']} | chunks={s['chunks']} | clean_chars={s['clean_chars']}")


if __name__ == "__main__":
    main()
