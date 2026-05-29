import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

SRC_V2 = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v2.jsonl"
SRC_V3 = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v3.jsonl"
GOLDEN = ROOT / "final_route/data/eval/stage3_golden_eval_v1.jsonl"

OUT = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v4.jsonl"
REPORT = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v4_report.json"

REQUIRED_SECTIONS = [
    "[상황 요약]",
    "[사고 유형]",
    "[사실관계 정리]",
    "[확인 필요]",
    "[관련 근거]",
    "[핵심 쟁점]",
    "[대처 방법]",
    "[주의]",
]

INSTRUCTION_V4 = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 근거, 핵심 쟁점, 대처 방법, 주의사항을 구조화하여 답변하라. "
    "반드시 [상황 요약], [사고 유형], [사실관계 정리], [확인 필요], [관련 근거], "
    "[핵심 쟁점], [대처 방법], [주의] 섹션 제목을 정확히 그대로 사용하라. "
    "섹션 제목의 띄어쓰기나 글자를 바꾸지 말라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라. "
    "반드시 자연스러운 한국어로만 답변하고, 영어 표현이나 깨진 문자, 외국어 문자를 섞지 말라. "
    "답변 마지막에는 [주의] 섹션까지 모두 작성한 뒤 반드시 <END>를 출력하라."
)

FORBIDDEN_SUBSTRINGS = [
    "무조건 100%",
    "상대 100%",
    "운전자 100%",
    "반드시 이깁니다",
    "확정입니다",
    "책임 없습니다",
    "신고 안 해도 됩니다",
    "합의 안 해도 됩니다",
    "무조건 처벌됩니다",
    "무조건 면책입니다",
    "CCTV footage",
    "Police report",
    "Accident report",
    "911",
]

BAD_KOREAN_PATTERNS = [
    "率", "ス", "が", "책им", "사ogo",
    "횡담", "횽단", "횃단", "횡断",
    "자전기", "자전구", "쌍점", "숾절",
    "블랙박ス", "速度", "책им지", "교통사ogo",
]

EXTRA_INPUTS = {
    "차랑 사람이 부딪혔는데 누가 잘못인지 알려줘.",
    "상대가 갑자기 끼어들었으니까 상대 100% 맞죠?",
    "눈 온 다음 날 빙판길에서 미끄러져 앞차를 박았어. 이건 자연재해니까 내 책임 없는 거 아니야?",
    "밤에 고라니가 튀어나와서 피하다가 중앙선을 넘고 반대편 차랑 부딪혔어.",
    "브레이크가 갑자기 안 들어서 앞차를 들이받았어. 이러면 고장이라서 괜찮은 거야?",
    "전동킥보드를 탄 채 횡단보도를 건너다가 차랑 부딪혔어. 자전거처럼 봐야 해?",
    "정차된 차 문이 갑자기 열리면서 지나가던 자전거가 부딪혔어.",
    "고속도로에서 사고가 났고 차가 멈췄어. 뒤에서 차들이 빠르게 오는데 어떻게 해야 해?",
}

def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

def make_text(instruction, user_input, output):
    return (
        f"### 지시문\n{instruction}\n\n"
        f"### 사용자 질문\n{user_input}\n\n"
        f"### 답변\n{output}"
    )

def golden_inputs():
    if not GOLDEN.exists():
        return set()
    return {row["input"].strip() for row in read_jsonl(GOLDEN)}

def clean_output_literal(output: str):
    # v4에서는 문장 구조를 크게 바꾸지 않고, 안전 검사에 걸리는 literal만 최소 치환한다.
    replacements = {
        "무조건 상대 100%라고 보면 되나?": "상대방 과실로 단정할 수 있는지 묻고 있다.",
        "상대 100%라고 단정할 수 없습니다.": "상대방에게 모든 책임이 있다고 단정할 수 없습니다.",
        "상대 100% 맞죠?": "상대방에게 모든 책임이 있는지 묻고 있다.",
    }
    for old, new in replacements.items():
        output = output.replace(old, new)

    if not output.rstrip().endswith("<END>"):
        output = output.rstrip() + "\n<END>"
    return output

def validate(row):
    out = row["output"]
    section_missing = [sec for sec in REQUIRED_SECTIONS if sec not in out]
    end_ok = out.rstrip().endswith("<END>")
    forbidden = [x for x in FORBIDDEN_SUBSTRINGS if x in out]
    bad_korean = [x for x in BAD_KOREAN_PATTERNS if x in out]
    return section_missing, end_ok, forbidden, bad_korean

def main():
    v2_rows = read_jsonl(SRC_V2)
    v3_rows = read_jsonl(SRC_V3)
    golden = golden_inputs()

    rows = []
    seen = set()

    # 1) v2는 안정성이 좋으므로 output을 거의 그대로 사용한다.
    for src in v2_rows:
        user_input = src["input"].strip()
        if user_input in golden or user_input in seen:
            continue

        output = clean_output_literal(src["output"])
        row = {
            "id": f"stage3_user_qa_v4_{len(rows)+1:05d}",
            "stage": "stage3_user_instruction",
            "task_type": "sft",
            "instruction": INSTRUCTION_V4,
            "input": user_input,
            "output": output,
            "text": make_text(INSTRUCTION_V4, user_input, output),
        }
        rows.append(row)
        seen.add(user_input)

    # 2) v3에서 만든 hard/edge case 8개만 추가한다.
    for src in v3_rows:
        user_input = src["input"].strip()
        if user_input not in EXTRA_INPUTS:
            continue
        if user_input in golden or user_input in seen:
            continue

        output = clean_output_literal(src["output"])
        row = {
            "id": f"stage3_user_qa_v4_{len(rows)+1:05d}",
            "stage": "stage3_user_instruction",
            "task_type": "sft",
            "instruction": INSTRUCTION_V4,
            "input": user_input,
            "output": output,
            "text": make_text(INSTRUCTION_V4, user_input, output),
        }
        rows.append(row)
        seen.add(user_input)

    section_fail = []
    end_fail = []
    forbidden_hits = []
    bad_korean_hits = []

    for row in rows:
        section_missing, end_ok, forbidden, bad_korean = validate(row)
        if section_missing:
            section_fail.append({"id": row["id"], "missing": section_missing})
        if not end_ok:
            end_fail.append(row["id"])
        if forbidden:
            forbidden_hits.append({"id": row["id"], "bad": forbidden})
        if bad_korean:
            bad_korean_hits.append({"id": row["id"], "bad": bad_korean})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8"
    )

    report = {
        "source_v2": str(SRC_V2),
        "source_v3_extra": str(SRC_V3),
        "output": str(OUT),
        "total": len(rows),
        "golden_excluded_count": len(golden),
        "section_fail": section_fail,
        "end_fail": end_fail,
        "forbidden_hits": forbidden_hits,
        "bad_korean_hits": bad_korean_hits,
        "input_dedup_count": len(seen),
        "avg_output_chars": round(sum(len(r["output"]) for r in rows) / max(len(rows), 1), 2),
        "stage_counts": dict(Counter(r["stage"] for r in rows)),
        "note": "v4 keeps v2 outputs stable and adds only selected v3 hard cases. No output compression is applied.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Stage3 v4 data generated]")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
