import json
import os

from run_final_roadsafe_inference import load_model, generate_answer


TEST_CASES = [
    {
        "name": "left_turn_vs_right_turn_bus",
        "question": "좌회전 신호를 받고 좌회전을 하는데 맞은편 차선에서 갑자기 우회전하는 버스랑 접촉 사고가 났어. 버스기사가 내려서 내 잘못이라고 욕하는데 어떻게 해?",
        "must_include": ["차 대 차", "좌회전", "우회전", "버스", "보험사"],
        "must_not_include": ["차 대 보행자", "보행자 신호위반", "무단횡단", "보행자 측 과실"],
    },
    {
        "name": "rear_end_collision",
        "question": "신호 대기 중인데 뒤차가 와서 내 차를 들이받았어. 상대가 내가 갑자기 멈췄다고 하는데 어떻게 해야 해?",
        "must_include": ["뒤차", "추돌", "블랙박스", "보험사"],
        "must_not_include": ["보행자 신호위반", "무단횡단", "자전거횡단도"],
    },
    {
        "name": "lane_change_contact",
        "question": "내가 2차로로 직진 중이었는데 옆 차가 방향지시등도 없이 끼어들다가 내 차 옆을 긁었어. 상대는 내가 양보 안 했다고 해.",
        "must_include": ["차선", "진로", "방향지시등", "보험사"],
        "must_not_include": ["보행자 신호위반", "무단횡단", "횡단 위치"],
    },
]


def check_case(result, case):
    final_answer = result["final_answer"]
    safety = result["safety_check"]

    missing = [kw for kw in case["must_include"] if kw not in final_answer]
    forbidden = [kw for kw in case["must_not_include"] if kw in final_answer]

    passed = safety.get("pass") and not missing and not forbidden

    return {
        "name": case["name"],
        "passed": passed,
        "missing": missing,
        "forbidden": forbidden,
        "fallback_used": safety.get("fallback_used"),
        "forced_fallback_reason": safety.get("forced_fallback_reason"),
        "semantic_mismatch": safety.get("semantic_mismatch"),
    }


def main():
    tokenizer, model = load_model()

    summary = []

    for case in TEST_CASES:
        print("\n" + "=" * 100)
        print(f"[TEST] {case['name']}")
        print(case["question"])

        result = generate_answer(tokenizer, model, case["question"])
        report = check_case(result, case)
        summary.append(report)

        if os.getenv("SHOW_FINAL", "0") == "1":
            print("\n[FINAL ANSWER]")
            print(result["final_answer"])

        print("\n[CHECK REPORT]")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    print("\n" + "=" * 100)
    print("[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
