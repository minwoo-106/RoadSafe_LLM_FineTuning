import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC_JSONL = ROOT / "final_route" / "data" / "stage1" / "roadsafe_stage1_law_dapt_v3.jsonl"

OUT_DIR = ROOT / "final_route" / "data" / "stage1"
OUT_JSONL = OUT_DIR / "roadsafe_stage1_law_dapt_v4.jsonl"
OUT_TXT = OUT_DIR / "roadsafe_stage1_law_dapt_v4.txt"
OUT_REPORT = OUT_DIR / "roadsafe_stage1_law_dapt_v4_report.json"

MAX_CHARS = 2200
MIN_CHARS = 450

BAD_PATTERNS = [
    "찾기쉬운생활법령",
    "--- PAGE",
    "\u0000",
    "····",
    "공공데이터정책",
    "법제처법제정보담당관",
    "https://www.easylaw.go.kr",
]


def fix_broken_years(text: str) -> str:
    # 예: 202\n0. 6. 9. -> 2020. 6. 9.
    text = re.sub(r"(20[0-9])\s*\n\s*([0-9]\.\s*\d{1,2}\.\s*\d{1,2}\.)", r"\1\2", text)
    text = re.sub(r"(19[0-9])\s*\n\s*([0-9]\.\s*\d{1,2}\.\s*\d{1,2}\.)", r"\1\2", text)

    # 예: 2\n011.3.10 -> 2011.3.10
    text = re.sub(r"\b(1|2)\s*\n\s*(0[0-9]{2}\.\s*\d{1,2}\.\s*\d{1,2})", r"\1\2", text)

    return text


def fix_common_spacing(text: str) -> str:
    # 생활법령 PDF에서 자주 붙는 표현 일부만 보수적으로 보정
    replacements = {
        "이정보는": "이 정보는",
        "안전하게도로주행하기": "안전하게 도로주행하기",
        "위반시제재": "위반 시 제재",
        "무면허운전": "무면허 운전",
        "음주운전": "음주 운전",
        "교통사고발생시": "교통사고 발생 시",
        "사고발생시": "사고 발생 시",
        "자동차운전": "자동차 운전",
        "운전면허": "운전면허",
        "자동차등": "자동차등",
        "자전거등": "자전거등",
        "도로교통법시행규칙": "도로교통법 시행규칙",
        "도로교통법시행령": "도로교통법 시행령",
        "특정범죄가중처벌등에관한법률": "특정범죄 가중처벌 등에 관한 법률",
        "교통사고처리특례법": "교통사고처리 특례법",
        "도로에서일어나는": "도로에서 일어나는",
        "교통상의위험": "교통상의 위험",
        "안전하고원활한": "안전하고 원활한",
        "교통을확보": "교통을 확보",
        "보도와차도": "보도와 차도",
        "차도가구분된": "차도가 구분된",
        "차도를통행": "차도를 통행",
        "중앙선침범": "중앙선 침범",
        "안전거리": "안전거리",
        "전방주시": "전방주시",
        "급제동": "급제동",
        "앞지르기": "앞지르기",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    return text


def remove_repeated_footer(text: str) -> str:
    # 예: 찾기쉬운생활법령 7 / 79 류 잔여 제거
    text = re.sub(r"찾기\s*쉬운\s*생활\s*법령\s*\d+\s*/\s*\d+", " ", text)
    text = re.sub(r"\n\s*\d+\s*/\s*\d+\s*\n", "\n", text)
    return text


def remove_excess_revision_noise(text: str) -> str:
    # 개정 이력이 문장 사이에 너무 자주 껴서 학습 품질을 흐리는 부분 완화
    # 조항 자체는 남기고 <개정 ...> 표기만 제거
    text = re.sub(r"<\s*개정[^>]*>", " ", text)
    text = re.sub(r"<\s*신설[^>]*>", " ", text)
    text = re.sub(r"<\s*삭제[^>]*>", " ", text)
    text = re.sub(r"\[\s*전문개정\s*[0-9.\s]+\]", " ", text)
    text = re.sub(r"\[\s*본조신설\s*[0-9.\s]+\]", " ", text)
    return text


def normalize_text(text: str) -> str:
    text = html.unescape(text)
    text = fix_broken_years(text)
    text = remove_repeated_footer(text)
    text = remove_excess_revision_noise(text)
    text = fix_common_spacing(text)

    # 이상하게 분리된 조 번호 보정: 제\n54조 -> 제54조
    text = re.sub(r"제\s*\n\s*(\d+조)", r"제\1", text)
    text = re.sub(r"제\s+(\d+조)", r"제\1", text)

    # 줄바꿈이 너무 자주 들어가 문장 흐름이 깨진 부분 완화
    # 단, 섹션/조항 번호 앞 줄바꿈은 유지하기 위해 먼저 보호
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 문장 중간 단순 줄바꿈은 공백으로 완화
    lines = [line.strip() for line in text.splitlines()]
    merged = []
    for line in lines:
        if not line:
            merged.append("")
            continue

        # 제목/조항/번호형 항목은 독립 줄 유지
        if re.match(r"^(제\d+조|제\d+장|제\d+절|\d+\.\s|\d+\.\d+\.|\[)", line):
            merged.append(line)
            continue

        if merged and merged[-1] and not merged[-1].endswith((".", ":", "다.", "니다.", "함.", "음.", "요.")):
            merged[-1] = merged[-1] + " " + line
        else:
            merged.append(line)

    text = "\n".join(merged)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)

    return text.strip()


def split_hard(text: str, max_chars: int):
    # 너무 긴 chunk가 생기면 문장 경계 기준으로 재분할
    sentence_pattern = (
        r"(?<=[.!?。])\s+|"
        r"(?<=다\.)\s+|"
        r"(?<=니다\.)\s+|"
        r"(?<=한다\.)\s+|"
        r"(?<=된다\.)\s+|"
        r"(?<=함\.)\s+"
    )

    parts = []
    for part in re.split(sentence_pattern, text):
        part = part.strip()
        if not part:
            continue
        if len(part) <= max_chars:
            parts.append(part)
        else:
            for i in range(0, len(part), max_chars):
                sub = part[i:i + max_chars].strip()
                if sub:
                    parts.append(sub)
    return parts


def repack_chunks(rows):
    new_rows = []
    buffer_by_source = {}

    # source별 순서를 유지하면서 너무 길어진 텍스트만 안전하게 분할
    for row in rows:
        text = normalize_text(row["text"])

        if any(b in text for b in BAD_PATTERNS):
            continue

        if len(text) < MIN_CHARS:
            continue

        parts = split_hard(text, MAX_CHARS) if len(text) > MAX_CHARS else [text]

        for part in parts:
            if len(part) < MIN_CHARS:
                continue

            new = dict(row)
            new["text"] = part[:MAX_CHARS].strip()
            new_rows.append(new)

    # id/chunk_index 재부여
    counters = {}
    final_rows = []
    for i, row in enumerate(new_rows, start=1):
        source = row["source_file"]
        counters[source] = counters.get(source, 0) + 1
        row["id"] = f"stage1_law_dapt_v4_{i:05d}"
        row["chunk_index"] = counters[source]
        final_rows.append(row)

    return final_rows


def main():
    rows = [json.loads(line) for line in SRC_JSONL.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = repack_chunks(rows)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    OUT_TXT.write_text(("\n\n" + "=" * 80 + "\n\n").join(r["text"] for r in rows), encoding="utf-8")

    lengths = [len(r["text"]) for r in rows]
    source_stats = {}
    for r in rows:
        source_stats.setdefault(r["source_file"], 0)
        source_stats[r["source_file"]] += 1

    report = {
        "source": str(SRC_JSONL),
        "output_jsonl": str(OUT_JSONL),
        "output_txt": str(OUT_TXT),
        "rows": len(rows),
        "total_text_chars": sum(lengths),
        "min_chars": min(lengths) if lengths else 0,
        "max_chars": max(lengths) if lengths else 0,
        "avg_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "source_chunks": source_stats,
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
    for k, v in source_stats.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
