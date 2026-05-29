import json
import re
from pathlib import Path

in_path = Path("data/processed/roadsafe_stage1_law_bridge_v2.jsonl")
out_path = Path("data/processed/roadsafe_stage1_law_bridge_v3_clean.jsonl")

def make_clean_summary(input_text: str) -> str:
    t = input_text

    if "고속도로" in t:
        return "고속도로 주행 중 차로 이용, 안전거리, 진로변경 또는 앞지르기와 관련된 사고·위험 상황입니다."
    if "급브레이크" in t or "급제동" in t or "추돌" in t:
        return "급제동 또는 후방추돌과 관련해 안전거리와 제동 사유를 함께 검토해야 하는 상황입니다."
    if "자전거횡단도" in t:
        return "자전거횡단도 통행 중 자동차와 충돌한 상황으로, 자전거 통행방법과 차량의 주의의무가 핵심입니다."
    if "자전거도로" in t:
        return "자전거도로 또는 자전거 통행 위치와 관련된 사고 상황입니다."
    if "자전거" in t:
        return "자전거와 자동차 또는 보행자 사이에 발생한 사고 상황입니다."
    if "전동킥보드" in t or "개인형 이동장치" in t or "PM" in t:
        return "개인형 이동장치 운행 중 통행방법, 신호 준수, 안전장비 여부가 문제되는 상황입니다."
    if "횡단보도" in t or "보행자" in t:
        return "횡단보도 또는 보행자 보호의무와 관련된 사고 상황입니다."
    if "차선" in t or "차로" in t or "깜빡이" in t or "진로" in t:
        return "차로 변경 또는 진로 변경 과정에서 충돌 위험이 발생한 상황입니다."
    if "사고가 났는데" in t or "접촉사고" in t or "현장" in t or "보험" in t:
        return "교통사고 발생 후 신고, 기록, 증거 보존 또는 보험 처리와 관련된 상황입니다."

    return "도로교통 사고 또는 안전 운전과 관련해 핵심 쟁점과 대처 방법을 검토해야 하는 상황입니다."

def replace_summary(output: str, new_summary: str) -> str:
    pattern = r"\[상황 요약\]\n.*?\n\n\[핵심 쟁점\]"
    repl = f"[상황 요약]\n{new_summary}\n\n[핵심 쟁점]"
    return re.sub(pattern, repl, output, flags=re.DOTALL)

rows = []
changed = 0

with in_path.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        row = json.loads(line)
        output = row["output"]

        if "구체적으로는" in output or "관련된 구체적으로는" in output:
            output = replace_summary(output, make_clean_summary(row["input"]))
            changed += 1

        row["output"] = output
        rows.append(row)

with out_path.open("w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"[OK] wrote {out_path}")
print(f"[OK] rows: {len(rows)}")
print(f"[OK] changed summaries: {changed}")
