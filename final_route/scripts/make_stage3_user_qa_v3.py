import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

SRC_V2 = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v2.jsonl"
GOLDEN = ROOT / "final_route/data/eval/stage3_golden_eval_v1.jsonl"

OUT = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v3.jsonl"
REPORT = ROOT / "final_route/data/stage3/roadsafe_stage3_user_qa_v3_report.json"

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

INSTRUCTION_V3 = (
    "사용자의 교통사고 상담 질문을 읽고 사고 상황을 구조화하여 답변하라. "
    "반드시 [상황 요약], [사고 유형], [사실관계 정리], [확인 필요], [관련 근거], "
    "[핵심 쟁점], [대처 방법], [주의] 섹션을 이 순서대로 포함하라. "
    "각 섹션은 짧고 명확하게 작성하고, 답변을 과도하게 늘리지 말라. "
    "반드시 자연스러운 한국어로만 답변하고 영어 표현, 깨진 문자, 외국어 문자를 섞지 말라. "
    "과실비율, 법적 책임, 처벌 가능성, 보험 결과를 단정하지 말라. "
    "정보가 부족하면 추가 확인 자료가 필요하다고 안내하라. "
    "답변 마지막에는 반드시 <END>를 출력하라."
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

def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        raise FileNotFoundError(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

def extract_sections(output: str):
    text = output.replace("<END>", "").strip()
    sections = {}
    for i, sec in enumerate(REQUIRED_SECTIONS):
        start = text.find(sec)
        if start == -1:
            sections[sec] = ""
            continue
        content_start = start + len(sec)
        next_positions = []
        for next_sec in REQUIRED_SECTIONS[i+1:]:
            pos = text.find(next_sec, content_start)
            if pos != -1:
                next_positions.append(pos)
        content_end = min(next_positions) if next_positions else len(text)
        sections[sec] = text[content_start:content_end].strip()
    return sections

def compact_lines(content: str, max_lines: int = 4, max_chars: int = 360):
    content = re.sub(r"\n{3,}", "\n\n", content.strip())
    lines = [ln.rstrip() for ln in content.splitlines() if ln.strip()]
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    compact = "\n".join(lines).strip()
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip() + "..."
    return compact

def fallback_for(sec: str, user_input: str):
    if sec == "[상황 요약]":
        return f"{user_input.strip()}"
    if sec == "[사고 유형]":
        return "교통사고 유형은 입력 내용만으로 단정하지 않고, 교통 주체와 장소를 기준으로 추가 확인이 필요합니다."
    if sec == "[사실관계 정리]":
        return "- 교통 주체, 신호, 진행 방향, 충돌 위치를 확인해야 합니다.\n- 속도, 시야, 도로 구조, 사고 직후 조치도 함께 정리해야 합니다."
    if sec == "[확인 필요]":
        return "- 블랙박스 원본\n- CCTV 또는 목격자 진술\n- 신호 상태와 충돌 위치\n- 현장 사진과 보험 접수 기록"
    if sec == "[관련 근거]":
        return "- 현재 단계에서는 일반적인 도로교통 사고 쟁점과 과실 판단 요소를 기준으로 안내합니다.\n- 정확한 법령, 판례, 과실비율 기준은 RAG 검색 결과와 함께 보강해야 합니다."
    if sec == "[핵심 쟁점]":
        return "- 사고 발생 경위와 각 당사자의 주의의무 위반 여부가 핵심입니다.\n- 현재 정보만으로 과실비율이나 책임을 단정하기 어렵습니다."
    if sec == "[대처 방법]":
        return "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n2. 현장 사진과 영상 자료를 보존합니다.\n3. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다."
    if sec == "[주의]":
        return "이 답변은 일반적인 교통사고 쟁점 안내입니다. 실제 과실비율과 법적 책임은 증거와 기관 판단에 따라 달라질 수 있으므로 단정하지 말아야 합니다."
    return "확인 필요"

def normalize_output(row):
    sections = extract_sections(row.get("output", ""))
    parts = []

    for sec in REQUIRED_SECTIONS:
        content = sections.get(sec, "")
        if not content:
            content = fallback_for(sec, row["input"])
        if sec in ["[상황 요약]", "[관련 근거]", "[주의]"]:
            content = compact_lines(content, max_lines=3, max_chars=300)
        elif sec in ["[사실관계 정리]", "[확인 필요]", "[핵심 쟁점]", "[대처 방법]"]:
            content = compact_lines(content, max_lines=4, max_chars=420)
        else:
            content = compact_lines(content, max_lines=2, max_chars=220)
        parts.append(f"{sec}\n{content}")

    return "\n\n".join(parts).strip() + "\n<END>"

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

def build_extra_cases():
    scenarios = [
        {
            "tag": "insufficient_info",
            "input": "차랑 사람이 부딪혔는데 누가 잘못인지 알려줘.",
            "summary": "차량과 보행자가 충돌했다는 정보만 있으며, 사고 장소와 신호, 진행 방향은 확인되지 않았다.",
            "atype": "차 대 보행자 / 정보 부족 사고",
            "facts": "- 차량과 보행자가 관련되어 있습니다.\n- 횡단보도 여부, 신호 상태, 충돌 위치, 속도는 확인되지 않았습니다.\n- 현재 정보만으로 과실이나 책임을 단정하기 어렵습니다.",
            "checks": "- 사고 장소와 횡단보도 여부\n- 차량 신호와 보행자 신호\n- 블랙박스, CCTV, 목격자 진술\n- 부상 정도와 사고 직후 조치",
            "issues": "- 보행자의 통행 위치와 차량의 전방주시 의무가 핵심입니다.\n- 정보가 부족하므로 과실비율 판단보다 사실관계 정리가 우선입니다.",
        },
        {
            "tag": "risky_100",
            "input": "상대가 갑자기 끼어들었으니까 상대 100% 맞죠?",
            "summary": "상대 차량의 급차로변경 또는 끼어들기 사고로 보이나, 구체적인 차로와 속도는 확인되지 않았다.",
            "atype": "차 대 차 / 차로변경 또는 끼어들기 사고",
            "facts": "- 상대 차량의 진로 변경이 쟁점입니다.\n- 내 차량의 속도, 안전거리, 회피 가능성도 함께 확인해야 합니다.\n- 상대 100%라고 단정할 수 없습니다.",
            "checks": "- 블랙박스 원본\n- 차로 표시와 방향지시등 사용 여부\n- 양 차량 속도와 충돌 부위\n- 사고 직전 거리와 회피 가능성",
            "issues": "- 진로변경 차량의 주의의무와 후행 차량의 방어운전 가능성이 함께 검토됩니다.\n- 과실비율은 증거와 세부 상황에 따라 달라질 수 있습니다.",
        },
        {
            "tag": "ice_road",
            "input": "눈 온 다음 날 빙판길에서 미끄러져 앞차를 박았어. 이건 자연재해니까 내 책임 없는 거 아니야?",
            "summary": "빙판길에서 미끄러져 앞차를 추돌한 상황이다.",
            "atype": "차 대 차 / 노면 결빙 추돌 사고",
            "facts": "- 노면 결빙과 추돌이 관련되어 있습니다.\n- 기상 상황, 감속 여부, 안전거리, 타이어 상태를 확인해야 합니다.\n- 자연재해라는 이유만으로 책임이 없다고 단정할 수 없습니다.",
            "checks": "- 당시 도로와 기상 상태\n- 차량 속도와 안전거리\n- 블랙박스 원본\n- 타이어 상태와 제동 흔적",
            "issues": "- 결빙 예견 가능성과 감속 의무가 핵심입니다.\n- 앞차와의 안전거리 유지 여부도 함께 검토됩니다.",
        },
        {
            "tag": "animal",
            "input": "밤에 고라니가 튀어나와서 피하다가 중앙선을 넘고 반대편 차랑 부딪혔어.",
            "summary": "야간에 야생동물을 피하는 과정에서 중앙선을 넘어 반대편 차량과 충돌한 상황이다.",
            "atype": "비정형 사고 / 야생동물 출현 및 중앙선 침범 충돌",
            "facts": "- 야생동물 출현, 야간 시야, 중앙선 침범, 반대편 차량 충돌이 관련됩니다.\n- 돌발성, 속도, 회피 가능성, 2차 사고 위험을 확인해야 합니다.",
            "checks": "- 블랙박스 원본\n- 야생동물 출현 시점과 거리\n- 차량 속도와 조향 경로\n- 도로 조명과 시야 상태",
            "issues": "- 돌발 장애물에 대한 회피 가능성과 중앙선 침범의 불가피성이 핵심입니다.\n- 사고 당시 속도와 전방주시 여부도 함께 검토됩니다.",
        },
        {
            "tag": "brake_failure",
            "input": "브레이크가 갑자기 안 들어서 앞차를 들이받았어. 이러면 고장이라서 괜찮은 거야?",
            "summary": "브레이크 이상을 주장하는 추돌 사고 상황이다.",
            "atype": "차 대 차 / 차량 고장 주장 추돌 사고",
            "facts": "- 브레이크 고장 주장과 앞차 추돌이 관련됩니다.\n- 정비 이력, 고장 발생 시점, 비상조치 여부를 확인해야 합니다.\n- 고장 주장만으로 책임이 면제된다고 단정할 수 없습니다.",
            "checks": "- 차량 정비 기록\n- 사고 직전 블랙박스와 제동 흔적\n- 고장 진단 자료\n- 비상등, 감속, 회피 조치 여부",
            "issues": "- 차량 관리 의무와 고장의 예견 가능성이 핵심입니다.\n- 실제 기계 결함인지 운전 조작 문제인지 확인해야 합니다.",
        },
        {
            "tag": "pm_crosswalk",
            "input": "전동킥보드를 탄 채 횡단보도를 건너다가 차랑 부딪혔어. 자전거처럼 봐야 해?",
            "summary": "전동킥보드를 탄 상태로 횡단보도를 건너다 차량과 충돌한 상황이다.",
            "atype": "차 대 PM / 횡단보도 통행 중 충돌",
            "facts": "- PM 탑승 상태, 횡단보도, 차량 충돌이 관련됩니다.\n- PM을 타고 있었는지 끌고 있었는지가 중요합니다.\n- 신호와 충돌 위치도 확인해야 합니다.",
            "checks": "- PM 탑승 또는 보행 상태\n- 횡단보도와 자전거횡단도 표시\n- 보행자 신호와 차량 신호\n- 블랙박스, CCTV, 현장 사진",
            "issues": "- PM의 통행 방법과 차량의 전방주시 의무가 핵심입니다.\n- 횡단보도에서의 법적 지위는 세부 사실에 따라 달라질 수 있습니다.",
        },
        {
            "tag": "parked_door",
            "input": "정차된 차 문이 갑자기 열리면서 지나가던 자전거가 부딪혔어.",
            "summary": "정차 차량의 문 개방 과정에서 지나가던 자전거와 충돌한 상황이다.",
            "atype": "차 대 자전거 / 개문 사고",
            "facts": "- 정차 차량 문 개방과 자전거 진행이 관련됩니다.\n- 문을 연 시점, 자전거 진행 위치, 회피 가능성을 확인해야 합니다.",
            "checks": "- 블랙박스 또는 주변 CCTV\n- 차량 정차 위치\n- 문 개방 시점과 각도\n- 자전거 진행 경로와 속도",
            "issues": "- 문을 여는 사람의 후방 확인 의무가 핵심입니다.\n- 자전거의 진행 위치와 회피 가능성도 함께 검토됩니다.",
        },
        {
            "tag": "emergency",
            "input": "고속도로에서 사고가 났고 차가 멈췄어. 뒤에서 차들이 빠르게 오는데 어떻게 해야 해?",
            "summary": "고속도로 사고 후 차량이 멈춰 2차 사고 위험이 있는 긴급 상황이다.",
            "atype": "긴급상황 / 고속도로 정차 및 2차 사고 위험",
            "facts": "- 고속도로 정차, 후속 차량 접근, 2차 사고 위험이 관련됩니다.\n- 법률 쟁점보다 안전 확보가 우선입니다.",
            "checks": "- 탑승자 부상 여부\n- 차량 이동 가능 여부\n- 갓길 또는 안전지대 위치\n- 경찰, 보험, 긴급출동 연락 여부",
            "issues": "- 2차 사고 예방과 탑승자 대피가 최우선입니다.\n- 사고 경위 판단은 안전 확보 후 증거를 바탕으로 정리해야 합니다.",
        },
    ]

    rows = []
    for idx, s in enumerate(scenarios, 1):
        output = (
            f"[상황 요약]\n{s['summary']}\n\n"
            f"[사고 유형]\n{s['atype']}\n\n"
            f"[사실관계 정리]\n{s['facts']}\n\n"
            f"[확인 필요]\n{s['checks']}\n\n"
            "[관련 근거]\n- 현재 단계에서는 일반적인 도로교통 사고 쟁점과 과실 판단 요소를 기준으로 안내합니다.\n- 정확한 법령, 판례, 과실비율 기준은 RAG 검색 결과와 함께 보강해야 합니다.\n\n"
            f"[핵심 쟁점]\n{s['issues']}\n\n"
            "[대처 방법]\n1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n2. 블랙박스, CCTV, 현장 사진 등 원본 자료를 보존합니다.\n3. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다.\n\n"
            "[주의]\n이 답변은 일반적인 교통사고 쟁점 안내입니다. 실제 과실비율과 법적 책임은 증거와 기관 판단에 따라 달라질 수 있으므로 단정하지 말아야 합니다.\n"
            "<END>"
        )
        rows.append({
            "id": f"stage3_user_qa_v3_extra_{idx:05d}",
            "stage": "stage3_user_instruction",
            "task_type": "sft",
            "instruction": INSTRUCTION_V3,
            "input": s["input"],
            "output": output,
            "text": make_text(INSTRUCTION_V3, s["input"], output),
        })
    return rows

def validate(row):
    out = row["output"]
    ok_sections = all(sec in out for sec in REQUIRED_SECTIONS)
    ok_end = out.rstrip().endswith("<END>")
    bad = [x for x in FORBIDDEN_SUBSTRINGS if x in out]
    return ok_sections, ok_end, bad

def main():
    v2_rows = read_jsonl(SRC_V2)
    golden = golden_inputs()

    rows = []
    seen_inputs = set()

    for src in v2_rows:
        user_input = src["input"].strip()
        if user_input in golden:
            continue
        if user_input in seen_inputs:
            continue
        output = normalize_output(src)
        row = {
            "id": f"stage3_user_qa_v3_{len(rows)+1:05d}",
            "stage": "stage3_user_instruction",
            "task_type": "sft",
            "instruction": INSTRUCTION_V3,
            "input": user_input,
            "output": output,
            "text": make_text(INSTRUCTION_V3, user_input, output),
        }
        rows.append(row)
        seen_inputs.add(user_input)

    for extra in build_extra_cases():
        if extra["input"] in golden or extra["input"] in seen_inputs:
            continue
        extra["id"] = f"stage3_user_qa_v3_{len(rows)+1:05d}"
        rows.append(extra)
        seen_inputs.add(extra["input"])

    section_fail = []
    end_fail = []
    forbidden_hits = []

    for row in rows:
        ok_sec, ok_end, bad = validate(row)
        if not ok_sec:
            section_fail.append(row["id"])
        if not ok_end:
            end_fail.append(row["id"])
        if bad:
            forbidden_hits.append({"id": row["id"], "bad": bad})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    report = {
        "source": str(SRC_V2),
        "output": str(OUT),
        "total": len(rows),
        "golden_excluded_count": len(golden),
        "section_fail": section_fail,
        "end_fail": end_fail,
        "forbidden_hits": forbidden_hits,
        "input_dedup_count": len(seen_inputs),
        "avg_output_chars": round(sum(len(r["output"]) for r in rows) / max(len(rows), 1), 2),
        "stage_counts": dict(Counter(r["stage"] for r in rows)),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Stage3 v3 data generated]")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
