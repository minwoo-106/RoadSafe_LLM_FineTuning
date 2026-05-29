import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_TEXT_DIR = ROOT / "data" / "processed" / "raw_texts"

OUT_DIR = ROOT / "final_route" / "data" / "stage1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "roadsafe_stage1_law_dapt_v3.jsonl"
OUT_TXT = OUT_DIR / "roadsafe_stage1_law_dapt_v3.txt"
OUT_REPORT = OUT_DIR / "roadsafe_stage1_law_dapt_v3_report.json"

INCLUDE_KEYWORDS = [
    "도로교통법",
    "교통_운전",
    "자전거_운전자",
]

EXCLUDE_KEYWORDS = [
    "긴급차량",
    "PM대자동차사고과실비율",
    "회전교차로사고_과실비율",
    "자동차사고_과실비율_인정기준",
]

MAX_CHARS = 2200
MIN_CHARS = 450

BAD_CHUNK_PATTERNS = [
    "찾기쉬운생활법령",
    "공공데이터정책",
    "법제처법제정보담당관",
    "위조ㆍ변조",
    "https://www.easylaw.go.kr",
    "저작권",
    "이정보는 2026년",
    "--- PAGE",
    "\u0000",
    "····",
]


def is_target_file(path: Path) -> bool:
    name = path.name
    if any(k in name for k in EXCLUDE_KEYWORDS):
        return False
    return any(k in name for k in INCLUDE_KEYWORDS)


def trim_source_intro(text: str, source_name: str) -> str:
    # 생활법령 PDF는 앞부분 고지/목차가 길어서 본문 시작점부터 사용
    if "교통_운전" in source_name:
        candidates = [
            "1. 자동차운전준비하기",
            "1. 자동차 운전 준비하기",
            "1.1. 운전자운전금지사항",
            "1.1. 운전자 운전금지사항",
        ]
        for c in candidates:
            idx = text.find(c)
            if idx > 0:
                return text[idx:]

    if "자전거_운전자" in source_name:
        candidates = [
            "1. 자전거 알아보기",
            "1. 자전거운전자",
            "1. 자전거 운전자",
            "1.1. 자전거",
        ]
        for c in candidates:
            idx = text.find(c)
            if idx > 0:
                return text[idx:]

    return text


def normalize_line(line: str) -> str:
    line = line.replace("\ufeff", " ")
    line = line.replace("\xa0", " ")
    line = line.replace("\u0000", " ")
    line = line.replace("\t", " ")
    line = line.replace("Ÿ", "- ")

    # 목차 점선/장식 제거
    line = re.sub(r"·{2,}", " ", line)
    line = re.sub(r"\.{4,}", " ", line)
    line = re.sub(r"[ ]{2,}", " ", line)

    return line.strip()


def is_noise_line(line: str) -> bool:
    if not line:
        return False

    if re.match(r"^--- PAGE \d+ ---$", line):
        return True

    if re.fullmatch(r"\d+\s*/\s*\d+", line):
        return True

    if re.fullmatch(r"-?\s*\d+\s*-?", line):
        return True

    if line in {"목 차", "목차", "차 례", "차례"}:
        return True

    noise_keywords = [
        "찾기쉬운생활법령",
        "공공데이터정책",
        "법제처법제정보담당관",
        "https://www.easylaw.go.kr",
        "이정보는 2026년",
        "이 정보는 2026년",
        "저작권",
        "위조ㆍ변조",
    ]

    if any(k in line for k in noise_keywords):
        return True

    # 기호 비율이 높은 목차/깨진 줄 제거
    if len(line) >= 20:
        symbols = sum(1 for ch in line if ch in "·.-_—–•")
        if symbols / len(line) > 0.30:
            return True

    return False


def clean_text(text: str, source_name: str) -> str:
    text = trim_source_intro(text, source_name)

    lines = []
    for raw_line in text.splitlines():
        line = normalize_line(raw_line)
        if is_noise_line(line):
            continue
        lines.append(line)

    text = "\n".join(lines)

    # 남은 페이지 마커 제거
    text = re.sub(r"--- PAGE \d+ ---", "\n", text)

    # 제목/조항 앞 줄바꿈 보강
    text = re.sub(r"(?<!\n)(제\s*\d+\s*장)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(제\s*\d+\s*절)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(제\s*\d+\s*조)", r"\n\1", text)

    # 생활법령 번호형 소제목 앞 줄바꿈
    text = re.sub(r"(?<!\n)(\d+\.\s*[^\n]{2,40})", r"\n\1", text)
    text = re.sub(r"(?<!\n)(\d+\.\d+\.\s*[^\n]{2,50})", r"\n\1", text)
    text = re.sub(r"(?<!\n)(Q\.)", r"\n\1", text)
    text = re.sub(r"(?<!\n)(A\.)", r"\n\1", text)

    # 지나친 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def split_paragraphs(text: str):
    raw_parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    refined = []

    for p in raw_parts:
        # 한 문단이 너무 길면 줄 단위로 다시 쪼갬
        if len(p) <= MAX_CHARS:
            refined.append(p)
        else:
            lines = [x.strip() for x in p.splitlines() if x.strip()]
            refined.extend(lines)

    return refined


def split_hard(text: str, max_chars: int):
    sentence_pattern = (
        r"(?<=[.!?。])\s+|"
        r"(?<=다\.)\s+|"
        r"(?<=니다\.)\s+|"
        r"(?<=한다\.)\s+|"
        r"(?<=된다\.)\s+|"
        r"(?<=함\.)\s+"
    )

    pieces = []
    for part in re.split(sentence_pattern, text):
        part = part.strip()
        if not part:
            continue

        if len(part) <= max_chars:
            pieces.append(part)
        else:
            # 그래도 길면 강제 분할
            for i in range(0, len(part), max_chars):
                sub = part[i:i + max_chars].strip()
                if sub:
                    pieces.append(sub)

    return pieces


def is_bad_chunk(chunk: str) -> bool:
    if len(chunk) < MIN_CHARS:
        return True

    if any(p in chunk for p in BAD_CHUNK_PATTERNS):
        return True

    # 한글 비율이 너무 낮으면 장비규격/표/깨진 텍스트 가능성
    hangul = len(re.findall(r"[가-힣]", chunk))
    if hangul / max(len(chunk), 1) < 0.25:
        return True

    return False


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
                if buf and not is_bad_chunk(buf):
                    chunks.append(buf.strip())
                buf = part

            while len(buf) > MAX_CHARS:
                head = buf[:MAX_CHARS].strip()
                if not is_bad_chunk(head):
                    chunks.append(head)
                buf = buf[MAX_CHARS:].strip()

    if buf and not is_bad_chunk(buf):
        chunks.append(buf.strip())

    return chunks


def main():
    source_files = sorted([p for p in RAW_TEXT_DIR.glob("*.txt") if is_target_file(p)])

    if not source_files:
        raise RuntimeError("No target source files found for Stage1 Law DAPT v3.")

    rows = []
    txt_blocks = []
    source_stats = []

    for src in source_files:
        raw = src.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw, src.name)
        paragraphs = split_paragraphs(cleaned)
        chunks = pack_chunks(paragraphs)

        source_stats.append({
            "source_file": src.name,
            "raw_chars": len(raw),
            "clean_chars": len(cleaned),
            "chunks": len(chunks),
        })

        for idx, chunk in enumerate(chunks, start=1):
            row = {
                "id": f"stage1_law_dapt_v3_{len(rows)+1:05d}",
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
