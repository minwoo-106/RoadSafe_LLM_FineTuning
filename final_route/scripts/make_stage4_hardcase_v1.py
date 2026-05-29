import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "final_route/data/stage4/roadsafe_stage4_hardcase_v1.jsonl"
REPORT = ROOT / "final_route/data/stage4/roadsafe_stage4_hardcase_v1_report.json"
GOLDEN = ROOT / "final_route/data/eval/stage3_golden_eval_v1.jsonl"

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

INSTRUCTION = (
    "사용자의 교통사고 상담 질문을 읽고, 사고 상황을 요약한 뒤 "
    "사고 유형, 사실관계, 확인 필요 자료, 관련 근거, 핵심 쟁점, 대처 방법을 구조화하여 답변하라. "
    "과실비율이나 법적 책임은 단정하지 말고, 정보가 부족한 경우 추가 확인이 필요하다고 안내하라. "
    "특히 급발진, 차량 고장, 빙판길, 야생동물, 고속도로 정차, 정보 부족, 상대방 100% 주장처럼 "
    "단정 위험이 큰 상황에서는 증거와 확인 필요 자료 중심으로 답변하라. "
    "답변 마지막에는 반드시 <END>를 출력하라."
)

FORBIDDEN = [
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
    "911",
    "CCTV footage",
    "Police report",
    "Accident report",
]

def read_jsonl(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

def golden_inputs():
    return {r["input"].strip() for r in read_jsonl(GOLDEN)}

def make_text(instruction, user_input, output):
    return (
        f"### 지시문\n{instruction}\n\n"
        f"### 사용자 질문\n{user_input}\n\n"
        f"### 답변\n{output}"
    )

def answer(summary, accident_type, facts, checks, issues, actions=None, caution=None):
    if actions is None:
        actions = [
            "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.",
            "2. 블랙박스, CCTV, 현장 사진 등 원본 자료를 보존합니다.",
            "3. 사고 당시 신호, 속도, 위치, 진행 방향을 시간순으로 정리합니다.",
            "4. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다.",
        ]
    if caution is None:
        caution = (
            "이 답변은 일반적인 교통사고 쟁점 안내입니다. 실제 과실비율과 법적 책임은 "
            "신호, 속도, 충돌 위치, 블랙박스, CCTV, 진술, 보험사·수사기관·법원 판단에 따라 "
            "달라질 수 있으므로 단정하지 말아야 합니다."
        )

    return (
        f"[상황 요약]\n{summary}\n\n"
        f"[사고 유형]\n{accident_type}\n\n"
        f"[사실관계 정리]\n{facts}\n\n"
        f"[확인 필요]\n{checks}\n\n"
        "[관련 근거]\n"
        "- 현재 단계에서는 일반적인 도로교통 사고 쟁점과 과실 판단 요소를 기준으로 안내합니다.\n"
        "- 정확한 법령, 판례, 과실비율 기준은 RAG 검색 결과와 함께 보강해야 합니다.\n\n"
        f"[핵심 쟁점]\n{issues}\n\n"
        f"[대처 방법]\n" + "\n".join(actions) + "\n\n"
        f"[주의]\n{caution}\n<END>"
    )

SCENARIOS = [
    {
        "tag": "insufficient_info",
        "base_inputs": [
            "차랑 사람이 부딪혔는데 누가 잘못인지 알려줘.",
            "사람이랑 차가 사고 났는데 과실 몇 대 몇이야?",
            "교통사고 났는데 상대가 잘못한 거 맞지?",
            "사고가 났는데 정보는 별로 없어. 누가 책임져야 해?",
        ],
        "summary": "차량과 사람 사이에 사고가 발생했지만, 장소와 신호, 진행 방향, 충돌 위치가 확인되지 않았다.",
        "atype": "정보 부족 사고 / 차 대 보행자 가능성",
        "facts": "- 차량과 보행자 또는 사람이 관련된 사고로 보입니다.\n- 횡단보도 여부, 신호 상태, 충돌 위치, 차량 속도는 확인되지 않았습니다.\n- 현재 정보만으로 과실비율이나 책임을 단정하기 어렵습니다.",
        "checks": "- 사고 장소와 횡단보도 여부\n- 차량 신호와 보행자 신호\n- 블랙박스, CCTV, 목격자 진술\n- 부상 정도와 사고 직후 조치",
        "issues": "- 보행자의 통행 위치와 차량의 전방주시 의무가 핵심입니다.\n- 정보가 부족한 상태에서는 과실비율보다 객관 자료 확보가 우선입니다.",
    },
    {
        "tag": "one_hundred_claim",
        "base_inputs": [
            "상대가 갑자기 끼어들었으니까 상대방이 전부 잘못한 거 맞죠?",
            "상대가 신호위반한 것 같은데 제가 책임질 일은 없죠?",
            "상대 차가 갑자기 들어왔으니 저는 과실 없는 거 아닌가요?",
            "상대방이 잘못했으니 저는 아무것도 안 해도 되나요?",
        ],
        "summary": "상대방의 신호위반 또는 급차로변경을 주장하는 사고 상황이다.",
        "atype": "차 대 차 / 단정 위험 사고",
        "facts": "- 상대방의 위반 행위가 주장되고 있습니다.\n- 내 차량의 속도, 안전거리, 전방주시, 회피 가능성도 함께 확인해야 합니다.\n- 상대방에게 모든 책임이 있다고 단정할 수 없습니다.",
        "checks": "- 블랙박스 원본\n- 신호 상태와 차로 표시\n- 방향지시등 사용 여부\n- 양 차량 속도, 충돌 부위, 사고 직전 거리",
        "issues": "- 상대방의 위반이 중요한 요소일 수 있지만, 내 차량의 주의의무도 함께 검토됩니다.\n- 과실비율은 영상과 현장 자료를 바탕으로 판단해야 합니다.",
    },
    {
        "tag": "ice_road",
        "base_inputs": [
            "빙판길에서 미끄러져 앞차를 박았는데 자연재해라서 괜찮은 거야?",
            "눈 온 다음 날 차가 미끄러져 사고 났어. 이건 내 책임 아니지?",
            "도로가 얼어서 브레이크가 안 먹고 추돌했어. 과실이 어떻게 돼?",
            "결빙 도로에서 미끄러진 사고는 운전자 책임이 없어?",
        ],
        "summary": "빙판길 또는 결빙 도로에서 미끄러져 추돌한 사고 상황이다.",
        "atype": "차 대 차 / 노면 결빙 추돌 사고",
        "facts": "- 노면 결빙과 추돌이 관련되어 있습니다.\n- 기상 상황, 감속 여부, 안전거리, 타이어 상태를 확인해야 합니다.\n- 자연재해라는 이유만으로 책임이 없다고 단정할 수 없습니다.",
        "checks": "- 당시 도로와 기상 상태\n- 차량 속도와 안전거리\n- 블랙박스 원본\n- 타이어 상태, 제동 흔적, 도로 관리 상태",
        "issues": "- 결빙 예견 가능성과 감속 의무가 핵심입니다.\n- 안전거리 유지 여부와 회피 가능성도 함께 검토됩니다.",
    },
    {
        "tag": "brake_failure",
        "base_inputs": [
            "브레이크가 갑자기 안 들어서 앞차를 들이받았어. 고장이면 괜찮은 거야?",
            "브레이크 고장 때문에 사고 났는데 제 책임이 없어질 수 있나요?",
            "차량 결함으로 추돌했다면 운전자 과실은 없는 거야?",
            "브레이크가 밀려서 사고 났다고 보험사에 말하면 되나?",
        ],
        "summary": "브레이크 이상 또는 차량 결함을 주장하는 추돌 사고 상황이다.",
        "atype": "차 대 차 / 차량 고장 주장 추돌 사고",
        "facts": "- 브레이크 고장 주장과 앞차 추돌이 관련되어 있습니다.\n- 정비 이력, 고장 발생 시점, 비상조치 여부를 확인해야 합니다.\n- 고장 주장만으로 책임이 면제된다고 단정할 수 없습니다.",
        "checks": "- 차량 정비 기록\n- 사고 직전 블랙박스와 제동 흔적\n- 고장 진단 자료\n- 비상등, 감속, 회피 조치 여부",
        "issues": "- 차량 관리 의무와 고장의 예견 가능성이 핵심입니다.\n- 실제 기계 결함인지 운전 조작 문제인지 확인해야 합니다.",
    },
    {
        "tag": "sudden_acceleration",
        "base_inputs": [
            "급발진 같아서 사고가 났는데 이러면 운전자 책임은 없는 거야?",
            "차가 갑자기 튀어나가서 사고 났어. 급발진이면 과실 없지?",
            "급발진 주장하려면 뭐부터 확인해야 해?",
            "페달을 안 밟았는데 차가 나가서 부딪혔어. 어떻게 정리해야 해?",
        ],
        "summary": "급발진 또는 차량 이상 작동을 주장하는 사고 상황이다.",
        "atype": "차량 이상 작동 주장 / 급발진 의심 사고",
        "facts": "- 급발진 주장이 있으나 현재 정보만으로 원인을 단정할 수 없습니다.\n- 페달 조작, 차량 기록, 블랙박스, EDR 등 객관 자료가 중요합니다.\n- 운전자 책임 유무는 증거와 조사 결과에 따라 달라질 수 있습니다.",
        "checks": "- 블랙박스 원본\n- EDR 또는 차량 기록 확인 가능성\n- 페달 조작 정황\n- 정비 이력, 사고 직전 속도 변화, 주변 CCTV",
        "issues": "- 차량 결함인지 운전 조작 문제인지가 핵심입니다.\n- 급발진 주장은 객관 자료가 부족하면 인정이 어려울 수 있으므로 증거 확보가 중요합니다.",
    },
    {
        "tag": "animal",
        "base_inputs": [
            "밤에 고라니가 튀어나와서 피하다가 중앙선을 넘고 반대편 차랑 부딪혔어.",
            "야생동물을 피하려다 사고가 났으면 제 책임이 줄어드나요?",
            "고라니 때문에 핸들을 꺾다가 사고 났어. 어떻게 봐야 해?",
            "동물이 갑자기 튀어나와서 중앙선을 넘었는데 어쩔 수 없던 거 아닌가요?",
        ],
        "summary": "야생동물을 피하는 과정에서 차로 이탈 또는 중앙선 침범 후 충돌한 상황이다.",
        "atype": "비정형 사고 / 야생동물 출현 및 회피 중 충돌",
        "facts": "- 야생동물 출현, 야간 시야, 회피 조작, 상대 차량 충돌이 관련됩니다.\n- 돌발성, 속도, 조명, 회피 가능성, 중앙선 침범의 불가피성을 확인해야 합니다.",
        "checks": "- 블랙박스 원본\n- 야생동물 출현 시점과 거리\n- 차량 속도와 조향 경로\n- 도로 조명, 시야 상태, 상대 차량 위치",
        "issues": "- 돌발 장애물에 대한 회피 가능성과 회피 방법의 적정성이 핵심입니다.\n- 중앙선 침범이나 2차 충돌이 발생했다면 불가피성 여부를 신중히 봐야 합니다.",
    },
    {
        "tag": "emergency_highway",
        "base_inputs": [
            "고속도로에서 사고가 났고 차가 멈췄어. 뒤에서 차들이 빠르게 오는데 어떻게 해야 해?",
            "고속도로 사고 후 차가 안 움직여. 먼저 뭘 해야 해?",
            "고속도로 한가운데 멈췄는데 보험사부터 불러야 해?",
            "고속도로에서 사고 난 뒤 차 안에 있어도 돼?",
        ],
        "summary": "고속도로 사고 후 차량이 멈춰 2차 사고 위험이 있는 긴급 상황이다.",
        "atype": "긴급상황 / 고속도로 정차 및 2차 사고 위험",
        "facts": "- 고속도로 정차, 후속 차량 접근, 2차 사고 위험이 관련됩니다.\n- 법률 쟁점보다 탑승자 안전 확보가 우선입니다.",
        "checks": "- 탑승자 부상 여부\n- 차량 이동 가능 여부\n- 갓길 또는 안전지대 위치\n- 경찰, 보험, 긴급출동 연락 여부",
        "issues": "- 2차 사고 예방과 탑승자 대피가 최우선입니다.\n- 사고 경위 판단은 안전 확보 후 증거를 바탕으로 정리해야 합니다.",
        "actions": [
            "1. 가능한 경우 비상등을 켜고 안전한 위치로 이동합니다.",
            "2. 차 안에 머무르는 것이 위험하면 가드레일 밖 등 안전지대로 대피합니다.",
            "3. 2차 사고 위험이 크면 경찰 또는 긴급출동에 연락합니다.",
            "4. 안전 확보 후 블랙박스와 현장 정보를 보존합니다.",
        ],
        "caution": "고속도로 사고 직후에는 과실 판단보다 생명과 2차 사고 예방이 우선입니다. 구체적인 책임 판단은 안전 확보 후 증거와 기관 판단에 따라 달라질 수 있습니다.",
    },
    {
        "tag": "door_open",
        "base_inputs": [
            "정차된 차 문이 갑자기 열리면서 지나가던 자전거가 부딪혔어.",
            "주차된 차 문이 열려서 자전거랑 사고 났는데 누구 책임이 커?",
            "문 열다가 자전거가 부딪혔다고 하는데 뭐부터 봐야 해?",
            "개문 사고는 무조건 문 연 사람이 잘못인가요?",
        ],
        "summary": "정차 또는 주차 차량의 문이 열리면서 지나가던 자전거와 충돌한 상황이다.",
        "atype": "차 대 자전거 / 개문 사고",
        "facts": "- 문 개방 시점과 자전거 진행 경로가 관련됩니다.\n- 문을 연 사람의 후방 확인 여부와 자전거의 속도, 진행 위치를 확인해야 합니다.",
        "checks": "- 블랙박스 또는 주변 CCTV\n- 차량 정차 위치\n- 문 개방 시점과 각도\n- 자전거 진행 경로와 속도",
        "issues": "- 문을 여는 사람의 후방 확인 의무가 핵심입니다.\n- 자전거의 진행 위치와 회피 가능성도 함께 검토됩니다.",
    },
]

def main():
    golden = golden_inputs()
    rows = []
    seen = set()

    for scenario in SCENARIOS:
        for user_input in scenario["base_inputs"]:
            if user_input in golden or user_input in seen:
                continue

            output = answer(
                summary=scenario["summary"],
                accident_type=scenario["atype"],
                facts=scenario["facts"],
                checks=scenario["checks"],
                issues=scenario["issues"],
                actions=scenario.get("actions"),
                caution=scenario.get("caution"),
            )

            row = {
                "id": f"stage4_hardcase_v1_{len(rows)+1:05d}",
                "stage": "stage4_hardcase_correction",
                "task_type": "sft",
                "category": scenario["tag"],
                "instruction": INSTRUCTION,
                "input": user_input,
                "output": output,
                "text": make_text(INSTRUCTION, user_input, output),
            }
            rows.append(row)
            seen.add(user_input)

    section_fail = []
    end_fail = []
    forbidden_hits = []

    for row in rows:
        out = row["output"]
        missing = [sec for sec in REQUIRED_SECTIONS if sec not in out]
        if missing:
            section_fail.append({"id": row["id"], "missing": missing})
        if not out.rstrip().endswith("<END>"):
            end_fail.append(row["id"])
        bad = [x for x in FORBIDDEN if x in out]
        if bad:
            forbidden_hits.append({"id": row["id"], "bad": bad})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )

    report = {
        "output": str(OUT),
        "total": len(rows),
        "golden_excluded_count": len(golden),
        "section_fail": section_fail,
        "end_fail": end_fail,
        "forbidden_hits": forbidden_hits,
        "category_counts": dict(Counter(r["category"] for r in rows)),
        "avg_output_chars": round(sum(len(r["output"]) for r in rows) / max(len(rows), 1), 2),
        "note": "Stage4 hard case correction data. Based on Stage3 full_v2 final candidate; only hard/risky cases are included.",
    }

    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[Stage4 hardcase v1 generated]")
    print(json.dumps(report, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
