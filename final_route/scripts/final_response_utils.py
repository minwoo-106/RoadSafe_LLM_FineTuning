import re
from typing import Dict, List, Tuple


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

SECTION_NORMALIZE_MAP = {
    "[상황 요 약]": "[상황 요약]",
    "[상황요약]": "[상황 요약]",
    "[상황 요축]": "[상황 요약]",
    "[상황 요추]": "[상황 요약]",

    "[사실관계정리]": "[사실관계 정리]",
    "[사실 관계 정리]": "[사실관계 정리]",

    "[확인필요]": "[확인 필요]",
    "[확인 자료]": "[확인 필요]",

    "[관련근거]": "[관련 근거]",
    "[관련 근 거]": "[관련 근거]",

    "[핵심쟁점]": "[핵심 쟁점]",
    "[핵심 쟁 점]": "[핵심 쟁점]",

    "[대처방법]": "[대처 방법]",
    "[D대처 방법]": "[대처 방법]",

    "[주의사항]": "[주의]",
    "[주의 사항]": "[주의]",
    "[NB]": "[주의]",
}

FORBIDDEN_PHRASES = [
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
]

BAD_KOREAN_PATTERNS = [
    "率", "ス", "が", "책им", "사ogo",
    "횡断", "블랙박ス", "速度", "교통사ogo",
    "ライト", "ジャ", "カー", "可能です",
    "Therefore", "please consult", "collision location",
    "blackbox", "witness statement",
]

ALLOWED_LATIN_WORDS = {
    "CCTV", "RAG", "PM", "END", "EDR", "AI", "API", "GPS", "ABS"
}

EMERGENCY_KEYWORDS = [
    "고속도로", "2차 사고", "2차사고", "차가 멈", "도로 한가운데",
    "불이", "화재", "피가", "출혈", "의식", "기절", "119",
    "구급차", "응급", "크게 다쳤", "사망", "갓길", "대피",
]


def strip_prompt_echo(text: str) -> str:
    """Remove prompt echoes and keep only answer-ish content."""
    if "### 답변" in text:
        text = text.split("### 답변", 1)[1]
    return text.strip()


def normalize_section_headers(text: str) -> str:
    for bad, good in SECTION_NORMALIZE_MAP.items():
        text = text.replace(bad, good)
    return text


def trim_at_end_token(text: str) -> str:
    if "<END>" in text:
        text = text.split("<END>", 1)[0].strip()
    return text.strip()


def extract_sections(text: str) -> Dict[str, str]:
    sections = {}
    for i, sec in enumerate(REQUIRED_SECTIONS):
        start = text.find(sec)
        if start == -1:
            sections[sec] = ""
            continue

        content_start = start + len(sec)
        next_positions = []
        for next_sec in REQUIRED_SECTIONS[i + 1:]:
            pos = text.find(next_sec, content_start)
            if pos != -1:
                next_positions.append(pos)

        content_end = min(next_positions) if next_positions else len(text)
        sections[sec] = text[content_start:content_end].strip()

    return sections


def default_section_content(section: str, user_input: str = "") -> str:
    if section == "[상황 요약]":
        return user_input.strip() if user_input.strip() else "입력된 사고 상황을 바탕으로 추가 확인이 필요합니다."

    if section == "[사고 유형]":
        return "교통사고 유형은 현재 정보만으로 단정하지 않고, 교통 주체와 장소를 기준으로 추가 확인이 필요합니다."

    if section == "[사실관계 정리]":
        return "- 사고 장소, 진행 방향, 신호 상태, 충돌 위치를 확인해야 합니다.\n- 차량 속도, 시야 제한, 사고 직후 조치도 함께 정리해야 합니다."

    if section == "[확인 필요]":
        return "- 블랙박스 원본\n- CCTV 또는 목격자 진술\n- 신호 상태와 충돌 위치\n- 현장 사진과 보험 접수 기록"

    if section == "[관련 근거]":
        return "- 현재 단계에서는 일반적인 도로교통 사고 쟁점과 과실 판단 요소를 기준으로 안내합니다.\n- 정확한 법령, 판례, 과실비율 기준은 RAG 검색 결과와 함께 보강해야 합니다."

    if section == "[핵심 쟁점]":
        return "- 사고 발생 경위와 각 당사자의 주의의무 위반 여부가 핵심입니다.\n- 현재 정보만으로 과실비율이나 법적 책임을 단정하기 어렵습니다."

    if section == "[대처 방법]":
        return "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n2. 블랙박스, CCTV, 현장 사진 등 원본 자료를 보존합니다.\n3. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다."

    if section == "[주의]":
        return "이 답변은 일반적인 교통사고 쟁점 안내입니다. 실제 과실비율과 법적 책임은 증거와 기관 판단에 따라 달라질 수 있으므로 단정하지 말아야 합니다."

    return "추가 확인이 필요합니다."


def compact_content(content: str, max_chars: int = 700) -> str:
    content = re.sub(r"\n{3,}", "\n\n", content.strip())
    content = re.sub(r"[ \t]{2,}", " ", content)

    if len(content) > max_chars:
        content = content[:max_chars].rstrip() + "..."

    return content.strip()


def format_response(raw_answer: str, user_input: str = "") -> str:
    """
    Final response formatter.
    - fixes common section header variants
    - rebuilds required section order
    - fills missing sections with safe defaults
    - ensures <END>
    """
    text = strip_prompt_echo(raw_answer)
    text = normalize_section_headers(text)
    text = trim_at_end_token(text)

    sections = extract_sections(text)

    # If the model produced almost no section headers, preserve raw answer as summary-ish context.
    found_count = sum(1 for sec in REQUIRED_SECTIONS if sections.get(sec))
    if found_count == 0 and text:
        sections["[상황 요약]"] = compact_content(text, max_chars=300)

    rebuilt = []
    for sec in REQUIRED_SECTIONS:
        content = sections.get(sec, "").strip()
        if not content:
            content = default_section_content(sec, user_input=user_input)

        # Keep sections readable, prevent runaway outputs.
        if sec in ("[상황 요약]", "[사고 유형]"):
            content = compact_content(content, max_chars=300)
        elif sec in ("[주의]", "[관련 근거]"):
            content = compact_content(content, max_chars=500)
        else:
            content = compact_content(content, max_chars=700)

        rebuilt.append(f"{sec}\n{content}")

    return "\n\n".join(rebuilt).strip() + "\n<END>"


def detect_forbidden(text: str) -> List[str]:
    return [phrase for phrase in FORBIDDEN_PHRASES if phrase in text]


def detect_bad_korean(text: str) -> List[str]:
    found = []

    for pattern in BAD_KOREAN_PATTERNS:
        if pattern in text:
            found.append(pattern)

    # Japanese kana detection
    if re.search(r"[\u3040-\u30ff]", text):
        found.append("JAPANESE_KANA")

    # CJK ideograph detection. Conservative: catches weird mixed output like 率/断/護.
    if re.search(r"[\u4e00-\u9fff]", text):
        found.append("CJK_IDEOGRAPH")

    # Latin word detection except allowed technical words.
    latin_words = re.findall(r"\b[A-Za-z]{2,}\b", text)
    suspicious = sorted({w for w in latin_words if w.upper() not in ALLOWED_LATIN_WORDS})
    if suspicious:
        found.append("LATIN_MIXED:" + ",".join(suspicious[:8]))

    return sorted(set(found))


def is_emergency_query(user_input: str) -> bool:
    return any(keyword in user_input for keyword in EMERGENCY_KEYWORDS)


def safety_check(answer: str) -> Dict[str, object]:
    forbidden = detect_forbidden(answer)
    bad_korean = detect_bad_korean(answer)
    missing_sections = [sec for sec in REQUIRED_SECTIONS if sec not in answer]
    has_end = answer.rstrip().endswith("<END>")

    return {
        "pass": not forbidden and not bad_korean and not missing_sections and has_end,
        "forbidden": forbidden,
        "bad_korean": bad_korean,
        "missing_sections": missing_sections,
        "has_end": has_end,
    }

def build_safe_fallback_response(user_input: str = "") -> str:
    """
    Deterministic safe fallback.
    Used when model output contains broken Korean, mixed foreign text, or forbidden phrases.
    """
    parts = []
    for sec in REQUIRED_SECTIONS:
        content = default_section_content(sec, user_input=user_input)
        parts.append(f"{sec}\n{content}")
    return "\n\n".join(parts).strip() + "\n<END>"



def classify_query_type(user_input: str) -> str:
    text = user_input.strip()

    if any(k in text for k in ["고속도로", "2차 사고", "2차사고", "도로 한가운데", "차가 멈", "갓길", "대피"]):
        return "emergency_highway"

    if any(k in text for k in ["100%", "전부 잘못", "과실 없는", "책임질 일은 없", "아무것도 안 해도"]):
        return "one_hundred_claim"

    if any(k in text for k in ["브레이크", "고장", "차량 결함", "급발진", "페달"]):
        return "vehicle_failure"

    if any(k in text for k in ["빙판", "눈 온", "결빙", "미끄러져", "얼어서"]):
        return "ice_road"

    if any(k in text for k in ["고라니", "야생동물", "동물", "중앙선"]):
        return "animal"

    if any(k in text for k in ["킥보드", "전동킥보드", "PM", "개인형 이동장치"]):
        return "pm_crosswalk"

    if any(k in text for k in ["자전거", "자전거횡단도", "자전거도로", "횡단보도"]):
        return "bicycle_or_crosswalk"

    if any(k in text for k in ["문이 열", "차 문", "개문", "정차한 차량 문", "주차된 차량 문"]):
        return "door_open"

    if any(k in text for k in ["누가 잘못", "과실 몇", "정보", "모르겠", "사고났는데"]):
        return "insufficient_info"

    return "general"


def build_typed_fallback_response(user_input: str = "") -> str:
    qtype = classify_query_type(user_input)

    summary = user_input.strip() if user_input.strip() else "입력된 사고 상황에 대해 추가 확인이 필요합니다."

    templates = {
        "emergency_highway": {
            "[상황 요약]": "고속도로 또는 도로 한가운데에서 사고 후 차량이 멈춰 2차 사고 위험이 있는 상황입니다.",
            "[사고 유형]": "긴급상황 / 사고 후 정차 및 2차 사고 위험",
            "[사실관계 정리]": "- 차량이 이동 가능한지 확인해야 합니다.\n- 탑승자 부상 여부와 후속 차량 접근 위험을 먼저 봐야 합니다.\n- 사고 원인보다 안전 확보가 우선입니다.",
            "[확인 필요]": "- 탑승자 부상 여부\n- 차량 이동 가능 여부\n- 갓길 또는 안전지대 위치\n- 경찰, 119, 보험사 연락 필요성",
            "[핵심 쟁점]": "- 2차 사고 예방과 탑승자 대피가 최우선입니다.\n- 과실 판단은 안전 확보 이후 블랙박스와 현장 자료를 바탕으로 검토해야 합니다.",
            "[대처 방법]": "1. 가능하면 비상등을 켜고 안전한 위치로 이동합니다.\n2. 차 안이 위험하면 가드레일 밖 등 안전지대로 대피합니다.\n3. 부상자나 2차 사고 위험이 있으면 119 또는 경찰 신고를 먼저 검토합니다.\n4. 안전 확보 후 블랙박스와 현장 정보를 보존합니다.",
        },
        "one_hundred_claim": {
            "[상황 요약]": "상대방에게 모든 책임이 있는지 묻는 사고 주장 상황입니다.",
            "[사고 유형]": "과실 다툼 / 단정 위험 사고",
            "[사실관계 정리]": "- 상대방의 끼어들기, 신호위반, 급차로변경 여부를 확인해야 합니다.\n- 내 차량의 속도, 안전거리, 전방주시, 회피 가능성도 함께 검토해야 합니다.\n- 한쪽 주장만으로 책임을 단정하기 어렵습니다.",
            "[확인 필요]": "- 블랙박스 원본\n- 차로와 노면표시\n- 방향지시등 사용 여부\n- 양 차량 속도, 충돌 부위, 사고 직전 거리",
            "[핵심 쟁점]": "- 상대방의 위반 행위가 중요한 요소일 수 있습니다.\n- 다만 내 차량의 주의의무와 회피 가능성도 함께 판단됩니다.\n- 과실비율은 증거와 기관 판단에 따라 달라질 수 있습니다.",
            "[대처 방법]": "1. 블랙박스 원본과 현장 사진을 보존합니다.\n2. 차로, 신호, 방향지시등, 충돌 위치를 시간순으로 정리합니다.\n3. 보험사에 접수하고 과실 산정 근거를 요청합니다.\n4. 단정 표현보다는 객관 자료 중심으로 설명합니다.",
        },
        "vehicle_failure": {
            "[상황 요약]": "브레이크 고장, 급발진, 차량 결함 등 차량 이상을 주장하는 사고 상황입니다.",
            "[사고 유형]": "차량 이상 작동 주장 / 차량 결함 또는 제동 문제 사고",
            "[사실관계 정리]": "- 차량 이상 주장이 있으나 현재 정보만으로 원인을 단정할 수 없습니다.\n- 정비 이력, 사고 직전 조작, 제동 흔적, 차량 기록을 확인해야 합니다.\n- 고장 주장만으로 책임이 면제된다고 보기 어렵습니다.",
            "[확인 필요]": "- 블랙박스 원본\n- 정비 이력과 고장 진단 자료\n- EDR 또는 차량 기록 확인 가능성\n- 페달 조작 정황, 제동 흔적, 사고 직전 속도 변화",
            "[핵심 쟁점]": "- 실제 기계 결함인지 운전 조작 문제인지가 핵심입니다.\n- 차량 관리 의무와 고장의 예견 가능성도 함께 검토됩니다.\n- 객관 자료 없이 급발진이나 고장을 단정하기 어렵습니다.",
            "[대처 방법]": "1. 차량을 임의로 수리하기 전 가능한 증거를 보존합니다.\n2. 블랙박스와 차량 기록 확인 가능성을 확보합니다.\n3. 정비소 진단 자료와 사고 당시 조작 정황을 정리합니다.\n4. 보험사와 필요 시 경찰에 사고 경위를 설명합니다.",
        },
        "ice_road": {
            "[상황 요약]": "빙판길 또는 결빙 도로에서 미끄러져 사고가 발생한 상황입니다.",
            "[사고 유형]": "노면 결빙 / 미끄럼 추돌 또는 차로 이탈 사고",
            "[사실관계 정리]": "- 노면 결빙, 감속 여부, 안전거리, 타이어 상태를 확인해야 합니다.\n- 자연재해라는 이유만으로 책임이 없다고 단정하기 어렵습니다.\n- 도로 상황을 예견할 수 있었는지도 중요합니다.",
            "[확인 필요]": "- 당시 기상과 도로 상태\n- 차량 속도와 안전거리\n- 블랙박스 원본\n- 타이어 상태, 제동 흔적, 도로 관리 상태",
            "[핵심 쟁점]": "- 결빙 예견 가능성과 감속 의무가 핵심입니다.\n- 안전거리 유지와 회피 가능성도 함께 검토됩니다.\n- 도로 관리 하자가 있었는지도 별도 확인이 필요합니다.",
            "[대처 방법]": "1. 사고 직후 도로 상태와 기상 상황을 사진으로 남깁니다.\n2. 블랙박스 원본과 제동 흔적 자료를 보존합니다.\n3. 차량 속도, 안전거리, 타이어 상태를 정리합니다.\n4. 보험사에 접수하고 필요하면 도로 관리 책임 여부도 확인합니다.",
        },
        "animal": {
            "[상황 요약]": "야생동물을 피하는 과정에서 차로 이탈, 중앙선 침범, 충돌이 발생한 상황입니다.",
            "[사고 유형]": "비정형 사고 / 야생동물 출현 및 회피 중 충돌",
            "[사실관계 정리]": "- 야생동물 출현 시점과 거리, 차량 속도, 조향 경로를 확인해야 합니다.\n- 돌발성이 있었는지, 회피가 불가피했는지가 중요합니다.\n- 중앙선 침범이나 2차 충돌이 있었다면 더 신중히 봐야 합니다.",
            "[확인 필요]": "- 블랙박스 원본\n- 야생동물 출현 시점과 거리\n- 차량 속도와 조향 경로\n- 도로 조명, 시야 상태, 상대 차량 위치",
            "[핵심 쟁점]": "- 돌발 장애물에 대한 회피 가능성과 회피 방법의 적정성이 핵심입니다.\n- 중앙선 침범이 불가피했는지 여부도 검토됩니다.",
            "[대처 방법]": "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n2. 블랙박스 원본과 사고 지점 사진을 보존합니다.\n3. 야생동물 출현 시점, 거리, 속도를 정리합니다.\n4. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다.",
        },
        "pm_crosswalk": {
            "[상황 요약]": "전동킥보드 또는 개인형 이동장치가 횡단보도 주변에서 차량과 충돌한 상황입니다.",
            "[사고 유형]": "차 대 PM / 횡단보도 또는 도로 통행 중 충돌",
            "[사실관계 정리]": "- PM을 타고 있었는지, 내려서 끌고 있었는지가 중요합니다.\n- 횡단보도, 자전거횡단도, 신호 상태를 확인해야 합니다.\n- 차량의 전방주시와 감속 여부도 함께 검토됩니다.",
            "[확인 필요]": "- 블랙박스 원본\n- PM 탑승 또는 보행 상태\n- 횡단보도와 자전거횡단도 표시\n- 보행자 신호와 차량 신호",
            "[핵심 쟁점]": "- PM의 통행 방법과 차량의 전방주시 의무가 핵심입니다.\n- 탑승 상태와 통행 위치에 따라 판단이 달라질 수 있습니다.",
            "[대처 방법]": "1. 부상 여부와 2차 사고 위험을 확인합니다.\n2. PM 탑승 상태와 횡단 위치를 증거로 남깁니다.\n3. 블랙박스, CCTV, 현장 사진을 보존합니다.\n4. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다.",
        },
        "bicycle_or_crosswalk": {
            "[상황 요약]": "자전거 또는 횡단보도 주변에서 차량과 충돌한 사고 상황입니다.",
            "[사고 유형]": "차 대 자전거 또는 차 대 보행자 / 횡단보도·자전거도로 관련 사고",
            "[사실관계 정리]": "- 자전거를 타고 있었는지, 내려서 끌고 있었는지 확인해야 합니다.\n- 자전거횡단도 표시와 신호 상태가 중요합니다.\n- 차량의 우회전, 직진, 전방주시 여부도 함께 검토해야 합니다.",
            "[확인 필요]": "- 블랙박스 원본\n- 횡단보도와 자전거횡단도 표시\n- 차량 신호와 보행자 신호\n- 충돌 위치, 차량 속도, 진행 방향",
            "[핵심 쟁점]": "- 자전거의 통행 방법과 차량의 주의의무가 핵심입니다.\n- 횡단 위치와 신호 상태에 따라 책임 판단이 달라질 수 있습니다.",
            "[대처 방법]": "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n2. 블랙박스 원본과 현장 사진을 보존합니다.\n3. 신호, 속도, 충돌 위치, 진행 방향을 시간순으로 정리합니다.\n4. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다.",
        },
        "door_open": {
            "[상황 요약]": "정차 또는 주차 차량의 문이 열리면서 지나가던 자전거 등과 충돌한 상황입니다.",
            "[사고 유형]": "차 대 자전거 / 개문 사고",
            "[사실관계 정리]": "- 문을 연 시점과 자전거 진행 경로가 중요합니다.\n- 문을 연 사람의 후방 확인 여부와 자전거 속도, 진행 위치를 확인해야 합니다.",
            "[확인 필요]": "- 블랙박스 또는 주변 CCTV\n- 차량 정차 위치\n- 문 개방 시점과 각도\n- 자전거 진행 경로와 속도",
            "[핵심 쟁점]": "- 문을 여는 사람의 후방 확인 의무가 핵심입니다.\n- 자전거의 진행 위치와 회피 가능성도 함께 검토됩니다.",
            "[대처 방법]": "1. 부상 여부를 확인하고 2차 사고를 방지합니다.\n2. 문 개방 상태, 차량 위치, 자전거 진행 경로를 사진으로 남깁니다.\n3. 블랙박스나 CCTV 확보 가능성을 확인합니다.\n4. 보험사에 접수하고 과실 산정 근거를 확인합니다.",
        },
        "insufficient_info": {
            "[상황 요약]": "사고가 발생했지만 책임 판단에 필요한 정보가 부족한 상황입니다.",
            "[사고 유형]": "정보 부족 사고 / 추가 확인 필요",
            "[사실관계 정리]": "- 사고 장소, 교통 주체, 신호 상태, 충돌 위치가 확인되어야 합니다.\n- 차량 속도, 진행 방향, 시야 제한, 사고 직후 조치도 함께 정리해야 합니다.",
            "[확인 필요]": "- 블랙박스 원본\n- CCTV 또는 목격자 진술\n- 신호 상태와 충돌 위치\n- 현장 사진, 진단서, 수리 견적",
            "[핵심 쟁점]": "- 현재 정보만으로 과실비율이나 법적 책임을 단정하기 어렵습니다.\n- 객관 자료를 확보한 뒤 사고 경위와 주의의무 위반 여부를 검토해야 합니다.",
            "[대처 방법]": "1. 부상 여부와 2차 사고 위험을 먼저 확인합니다.\n2. 블랙박스, CCTV, 현장 사진을 보존합니다.\n3. 사고 당시 신호, 속도, 위치, 진행 방향을 시간순으로 정리합니다.\n4. 보험사에 접수하고 필요하면 경찰 신고를 검토합니다.",
        },
        "general": {
            "[상황 요약]": summary,
            "[사고 유형]": "교통사고 유형은 현재 정보만으로 단정하지 않고, 교통 주체와 장소를 기준으로 추가 확인이 필요합니다.",
            "[사실관계 정리]": default_section_content("[사실관계 정리]", user_input),
            "[확인 필요]": default_section_content("[확인 필요]", user_input),
            "[핵심 쟁점]": default_section_content("[핵심 쟁점]", user_input),
            "[대처 방법]": default_section_content("[대처 방법]", user_input),
        },
    }

    selected = templates.get(qtype, templates["general"])

    parts = []
    for sec in REQUIRED_SECTIONS:
        if sec == "[관련 근거]":
            content = "- 현재 단계에서는 일반적인 도로교통 사고 쟁점과 과실 판단 요소를 기준으로 안내합니다.\n- 정확한 법령, 판례, 과실비율 기준은 RAG 검색 결과와 함께 보강해야 합니다."
        elif sec == "[주의]":
            content = "이 답변은 일반적인 교통사고 쟁점 안내입니다. 실제 과실비율과 법적 책임은 증거와 기관 판단에 따라 달라질 수 있으므로 단정하지 말아야 합니다."
        else:
            content = selected.get(sec, default_section_content(sec, user_input=user_input))

        parts.append(f"{sec}\n{content}")

    return "\n\n".join(parts).strip() + "\n<END>"


def emergency_prefix(user_input: str) -> str:
    if not is_emergency_query(user_input):
        return ""

    return (
        "[긴급 안전 안내]\n"
        "2차 사고나 부상 위험이 있는 상황이면 과실 판단보다 안전 확보가 우선입니다. "
        "가능하면 안전지대로 대피하고, 부상자나 화재·고속도로 정차 등 위험이 있으면 119 또는 경찰 신고를 먼저 검토하세요.\n\n"
    )


def finalize_answer(raw_answer: str, user_input: str = "") -> Tuple[str, Dict[str, object]]:
    formatted = format_response(raw_answer, user_input=user_input)
    check = safety_check(formatted)

    # If the model output is unsafe or linguistically broken, replace it with a deterministic fallback.
    if check["forbidden"] or check["bad_korean"]:
        formatted = build_typed_fallback_response(user_input=user_input)
        check = safety_check(formatted)
        check["fallback_used"] = True
    else:
        check["fallback_used"] = False

    prefix = emergency_prefix(user_input)
    if prefix and not formatted.startswith("[긴급 안전 안내]"):
        formatted = prefix + formatted
        check = safety_check(formatted)
        check["fallback_used"] = check.get("fallback_used", False)

    return formatted, check


if __name__ == "__main__":
    sample_input = "사고났는데 누가 잘못이야?"
    sample_raw = """
[상황요약]
사고가 났지만 정보가 부족합니다.

[사고 유형]
정보 부족 사고

[대처방법]
보험사에 접수합니다.
"""
    final, check = finalize_answer(sample_raw, sample_input)
    print(final)
    print("\n[CHECK]")
    print(check)

    assert "[상황 요약]" in final
    assert "[사실관계 정리]" in final
    assert "[대처 방법]" in final
    assert final.rstrip().endswith("<END>")
    print("\n[OK] final_response_utils self-test passed")
