import json
from pathlib import Path
from collections import Counter

from run_final_roadsafe_inference import load_model, generate_answer
from final_response_utils import detect_bad_korean, detect_forbidden


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "final_route/reports/final_eval"
OUT_JSONL = OUT_DIR / "final_road_sign_eval_v2_results.jsonl"
OUT_REPORT = OUT_DIR / "final_road_sign_eval_v2_report.json"


QUESTIONS = [
    {
        "id": "sign_001",
        "category": "rockfall_vehicle",
        "question": "낙석주의 표지판이 있는 산길에서 돌이 굴러와서 내 차 앞부분이 파손됐어.",
        "must_include_any": ["낙석", "표지판", "도로관리", "블랙박스", "단정"],
    },
    {
        "id": "sign_002",
        "category": "rockfall_pedestrian",
        "question": "낙석주의 표지판 있는 보행로 옆을 걷다가 작은 돌에 맞았어. 누구 책임이야?",
        "must_include_any": ["낙석", "보행자", "도로관리", "사진", "단정"],
    },
    {
        "id": "sign_003",
        "category": "slippery_road",
        "question": "노면 미끄럼 주의 표지판 있는 곳에서 비 때문에 미끄러져 옆 차랑 부딪혔어.",
        "must_include_any": ["노면", "미끄럼", "감속", "안전거리", "블랙박스"],
    },
    {
        "id": "sign_004",
        "category": "animal_warning",
        "question": "야생동물 출현 주의 표지판 있는 도로에서 고라니가 튀어나와서 사고 났어.",
        "must_include_any": ["야생동물", "고라니", "표지판", "회피", "블랙박스"],
    },
    {
        "id": "sign_005",
        "category": "falling_rocks_no_sign",
        "question": "낙석주의 표지판은 못 봤는데 산길에서 돌이 떨어져서 차가 망가졌어.",
        "must_include_any": ["낙석", "표지판", "도로관리", "사진", "확인"],
    },
    {
        "id": "sign_006",
        "category": "construction_sign",
        "question": "공사중 표지판 보고 서행했는데 임시 차선이 헷갈려서 옆 차랑 부딪혔어.",
        "must_include_any": ["공사중", "임시 차선", "표지", "블랙박스", "확인"],
    },
    {
        "id": "sign_007",
        "category": "lane_reduction",
        "question": "차로 감소 표지판 있는 곳에서 합류하다가 옆 차랑 접촉했어.",
        "must_include_any": ["차로 감소", "합류", "차로", "블랙박스", "단정"],
    },
    {
        "id": "sign_008",
        "category": "school_zone_sign",
        "question": "어린이보호구역 표지판 있는 곳에서 아이가 갑자기 뛰어나왔어.",
        "must_include_any": ["어린이보호구역", "어린이", "속도", "전방주시", "블랙박스"],
    },
    {
        "id": "sign_009",
        "category": "speed_bump",
        "question": "과속방지턱 표지판을 못 보고 지나가다 차 하부가 긁혔어.",
        "must_include_any": ["과속방지턱", "표지", "속도", "도로관리", "단정"],
    },
    {
        "id": "sign_010",
        "category": "crosswalk_sign",
        "question": "횡단보도 표지판 있는 곳에서 보행자가 갑자기 뛰어나왔는데 내가 더 불리해?",
        "must_include_any": ["횡단보도", "보행자", "전방주시", "속도", "단정"],
    },
    {
        "id": "sign_011",
        "category": "bicycle_road_sign",
        "question": "자전거도로 표지 있는 곳에서 우회전하다 자전거랑 부딪혔어.",
        "must_include_any": ["자전거도로", "우회전", "자전거", "블랙박스", "확인"],
    },
    {
        "id": "sign_012",
        "category": "one_way",
        "question": "일방통행 표지판을 못 보고 들어갔다가 마주 오던 차랑 접촉했어.",
        "must_include_any": ["일방통행", "표지판", "진입", "블랙박스", "단정"],
    },
    {
        "id": "sign_013",
        "category": "no_u_turn",
        "question": "유턴금지 표지판 있는 곳에서 유턴하다가 오토바이랑 사고 났어.",
        "must_include_any": ["유턴금지", "유턴", "오토바이", "표지판", "블랙박스"],
    },
    {
        "id": "sign_014",
        "category": "no_parking",
        "question": "주정차금지 표지 있는 곳에 잠깐 세웠는데 자전거가 피하다 넘어졌어.",
        "must_include_any": ["주정차금지", "자전거", "정차", "위치", "확인"],
    },
    {
        "id": "sign_015",
        "category": "falling_object_truck",
        "question": "낙하물 주의 표지 있는 고속도로에서 앞 화물차 짐이 떨어져 사고 났어.",
        "must_include_any": ["낙하물", "화물차", "고속도로", "블랙박스", "적재물"],
    },
    {
        "id": "sign_016",
        "category": "curve_warning",
        "question": "급커브 주의 표지판 있는 길에서 속도를 못 줄여 가드레일을 긁었어.",
        "must_include_any": ["급커브", "속도", "가드레일", "표지판", "단정"],
    },
    {
        "id": "sign_017",
        "category": "steep_downhill",
        "question": "급경사 내리막 표지판 있는 길에서 브레이크가 밀려 앞차를 박았어.",
        "must_include_any": ["급경사", "내리막", "브레이크", "안전거리", "블랙박스"],
    },
    {
        "id": "sign_018",
        "category": "road_closed",
        "question": "통행금지 표지판이 있었는데 길이 열려 보여서 들어갔다가 사고 났어.",
        "must_include_any": ["통행금지", "표지판", "진입", "확인", "단정"],
    },
    {
        "id": "sign_019",
        "category": "merge_sign",
        "question": "합류도로 표지판 있는 곳에서 본선 차랑 부딪혔어. 누가 양보해야 해?",
        "must_include_any": ["합류", "본선", "양보", "블랙박스", "단정"],
    },
    {
        "id": "sign_020",
        "category": "pedestrian_warning",
        "question": "보행자주의 표지판 있는 도로에서 사람이 갑자기 차도로 내려왔어.",
        "must_include_any": ["보행자주의", "보행자", "전방주시", "속도", "블랙박스"],
    },
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer, model = load_model()

    results = []

    for idx, item in enumerate(QUESTIONS, 1):
        print(f"[{idx}/{len(QUESTIONS)}] {item['id']} | {item['category']}")

        result = generate_answer(tokenizer, model, item["question"])
        raw = result["raw_answer"]
        final = result["final_answer"]
        check = result["safety_check"]

        raw_bad = detect_bad_korean(raw)
        raw_forbidden = detect_forbidden(raw)

        must_include_any = item.get("must_include_any", [])
        expected_hits = [kw for kw in must_include_any if kw in final]
        expectation_pass = len(expected_hits) > 0 if must_include_any else True

        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "final_pass": check.get("pass"),
            "fallback_used": check.get("fallback_used"),
            "forced_fallback_reason": check.get("forced_fallback_reason"),
            "raw_bad": raw_bad,
            "raw_forbidden": raw_forbidden,
            "final_bad_korean": check.get("bad_korean"),
            "final_forbidden": check.get("forbidden"),
            "missing_sections": check.get("missing_sections"),
            "has_end": check.get("has_end"),
            "must_include_any": must_include_any,
            "expected_hits": expected_hits,
            "expectation_pass": expectation_pass,
            "raw_answer": raw,
            "final_answer": final,
        })

    OUT_JSONL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
        encoding="utf-8"
    )

    total = len(results)
    report = {
        "total": total,
        "final_pass": sum(1 for r in results if r["final_pass"]),
        "fallback_used": sum(1 for r in results if r["fallback_used"]),
        "fallback_rate": round(sum(1 for r in results if r["fallback_used"]) / total, 4),
        "raw_bad_clean": sum(1 for r in results if not r["raw_bad"]),
        "final_bad_clean": sum(1 for r in results if not r["final_bad_korean"]),
        "forbidden_clean": sum(1 for r in results if not r["final_forbidden"]),
        "has_end": sum(1 for r in results if r["has_end"]),
        "section_full": sum(1 for r in results if not r["missing_sections"]),
        "expectation_pass": sum(1 for r in results if r["expectation_pass"]),
        "fallback_reasons": dict(Counter(r["forced_fallback_reason"] for r in results if r["forced_fallback_reason"])),
        "output_jsonl": str(OUT_JSONL),
    }

    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[FINAL ROAD SIGN EVAL V2 REPORT]")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
