import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "final_route" / "data" / "stage2" / "roadsafe_stage2_accident_structure_v1.jsonl"
OUT_DIR = ROOT / "final_route" / "data" / "stage3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSONL = OUT_DIR / "roadsafe_stage3_user_qa_v1.jsonl"
OUT_REPORT = OUT_DIR / "roadsafe_stage3_user_qa_v1_report.json"

STAGE3_INSTRUCTION = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 쟁점, 대처 방법을 구조화하여 답변하라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라."
)

REQUIRED_SECTIONS = [
    "[상황 요약]",
    "[사고 유형]",
    "[사실관계 정리]",
    "[확인 필요]",
    "[관련 근거]",
    "[핵심 쟁점]",
    "[대처 방법]",
    "[주의]",
    "<END>",
]


def normalize(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("CCTV footage", "CCTV 영상")
    s = s.replace("Accident report", "사고 접수 자료")
    s = s.replace("Police report", "경찰 신고 또는 사고 접수 자료")
    s = s.replace("Police investigation", "경찰 조사 자료")
    s = s.replace("Witness statements", "목격자 진술")
    s = s.replace("Call the police immediately.", "필요하면 즉시 112에 신고합니다.")
    s = s.replace("911", "112 또는 119")
    s = s.replace("[핵심 판단요소]", "[핵심 판단 요소]")
    s = s.replace("[참고 과실 쌍점]", "[참고 과실 쟁점]")
    s = s.replace("[참고 과실 숾절]", "[참고 과실 쟁점]")
    s = s.replace("[추가 확인자료]", "[추가 확인 자료]")
    s = s.replace("[대처방법]", "[대처 방법]")
    s = s.replace("교통사망", "교통사고")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def section(text: str, name: str) -> str:
    pattern = rf"\[{re.escape(name)}\]\n(.*?)(?=\n\[[^\]]+\]|\Z)"
    m = re.search(pattern, text, flags=re.S)
    return m.group(1).strip() if m else ""


def make_user_variants(accident: str):
    accident = accident.strip().rstrip(".")
    return [
        f"{accident}. 이런 사고는 뭐부터 확인해야 해?",
        f"{accident}. 과실이나 대처 방법이 궁금합니다.",
        f"{accident}. 보험사에 말하기 전에 사고 쟁점을 정리해줘.",
    ]


def infer_subjects(accident: str):
    car = "확인 필요"
    other = "확인 필요"

    if any(k in accident for k in ["자동차", "차량", "앞차", "내 차", "승용차", "차"]):
        car = "관련 있음"

    if "자전거" in accident:
        other = "자전거"
    elif any(k in accident for k in ["전동킥보드", "PM", "개인형 이동장치"]):
        other = "개인형 이동장치(PM)"
    elif "보행자" in accident or "무단횡단" in accident:
        other = "보행자"
    elif "오토바이" in accident or "이륜차" in accident:
        other = "이륜차"
    elif any(k in accident for k in ["앞차", "뒤차", "차로", "급제동", "추돌"]):
        other = "다른 차량"

    return car, other


def build_answer(accident: str, stage2_output: str) -> str:
    stage2_output = normalize(stage2_output)

    accident_type = section(stage2_output, "사고 유형")
    factors = section(stage2_output, "핵심 판단 요소")
    issues = section(stage2_output, "참고 과실 쟁점")
    evidence = section(stage2_output, "추가 확인 자료")
    actions = section(stage2_output, "대처 방법")

    car, other = infer_subjects(accident)

    if not accident_type:
        accident_type = "확인 필요"

    if not factors:
        factors = "- 사고 위치\n- 각 교통주체의 진행 방향\n- 신호와 속도\n- 충돌 위치\n- 블랙박스/CCTV 여부"

    if not issues:
        issues = (
            "- 현재 정보만으로 과실비율을 단정하기 어렵습니다.\n"
            "- 신호, 속도, 충돌 위치, 전방주시, 회피 가능성 등 수정 요소를 함께 확인해야 합니다."
        )

    if not evidence:
        evidence = "- 블랙박스 원본\n- CCTV\n- 현장 사진\n- 신호 상태\n- 충돌 위치"

    if not actions:
        actions = (
            "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n"
            "2. 블랙박스 원본과 현장 사진을 보존합니다.\n"
            "3. 신호, 속도, 충돌 위치, 진행 방향을 시간순으로 정리합니다.\n"
            "4. 보험사에 접수하고 필요하면 경찰 신고 또는 전문가 상담을 검토합니다."
        )

    answer = f"""[상황 요약]
{accident}

[사고 유형]
{accident_type}

[사실관계 정리]
- 자동차: {car}
- 상대 교통주체: {other}
- 충돌 지점: 확인 필요
- 위험 요인: 신호, 속도, 진행 방향, 충돌 위치, 시야 제한 여부 확인 필요

[확인 필요]
{evidence}

[관련 근거]
- 현재 단계에서는 사고 유형별 일반 교통사고 쟁점과 과실 판단 요소를 기준으로 안내합니다.
- 정확한 법령 조항과 판례·과실비율 근거는 RAG 검색 결과와 함께 최종 답변에서 보강해야 합니다.

[핵심 쟁점]
{issues}

[대처 방법]
{actions}

[주의]
이 답변은 일반적인 교통사고·도로교통 쟁점 안내입니다. 실제 과실비율과 법적 책임은 신호, 속도, 충돌 위치, 블랙박스, CCTV, 진술, 보험사·수사기관·법원 판단에 따라 달라질 수 있습니다. 따라서 특정 비율로 단정하지 말아야 합니다.
<END>"""

    return normalize(answer)


def main():
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    src_rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    out_rows = []
    bad_rows = []

    for src in src_rows:
        accident = normalize(src["input"])
        stage2_output = normalize(src["output"])

        for user_input in make_user_variants(accident):
            answer = build_answer(accident, stage2_output)

            row = {
                "id": f"stage3_user_qa_v1_{len(out_rows)+1:05d}",
                "stage": "stage3_user_instruction",
                "task_type": "sft",
                "instruction": STAGE3_INSTRUCTION,
                "input": user_input,
                "output": answer,
                "text": (
                    "### 지시문\n"
                    f"{STAGE3_INSTRUCTION}\n\n"
                    "### 사용자 질문\n"
                    f"{user_input}\n\n"
                    "### 답변\n"
                    f"{answer}"
                ),
            }

            missing = [sec for sec in REQUIRED_SECTIONS if sec not in answer]
            if missing:
                bad_rows.append({
                    "id": row["id"],
                    "missing": missing,
                })

            out_rows.append(row)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "source": str(SRC),
        "output": str(OUT_JSONL),
        "source_rows": len(src_rows),
        "output_rows": len(out_rows),
        "bad_rows": len(bad_rows),
        "bad_row_details": bad_rows[:20],
        "required_sections": REQUIRED_SECTIONS,
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] wrote {OUT_JSONL}")
    print(f"[OK] wrote {OUT_REPORT}")
    print("[SOURCE_ROWS]", len(src_rows))
    print("[OUTPUT_ROWS]", len(out_rows))
    print("[BAD_ROWS]", len(bad_rows))


if __name__ == "__main__":
    main()
