from run_final_roadsafe_inference import load_model, generate_answer

questions = [
    "자전거를 탄 채 일반 횡단보도를 건너다가 우회전하던 자동차와 부딪혔다. 자전거횡단도 표시는 없었다.",
    "상대가 갑자기 끼어들었으니까 상대방 100% 맞지?",
    "고속도로에서 사고 후 차가 도로 한가운데 멈췄는데 2차 사고가 날까 봐 무섭다.",
    "사고났는데 누가 잘못이야?",
    "브레이크가 갑자기 안 들어서 앞차를 들이받았어. 고장이면 괜찮은 거야?",
]

tokenizer, model = load_model()

for i, q in enumerate(questions, 1):
    result = generate_answer(tokenizer, model, q)
    check = result["safety_check"]

    print("=" * 100)
    print(f"[{i}] {q}")
    print("pass:", check.get("pass"))
    print("fallback_used:", check.get("fallback_used"))
    print("forbidden:", check.get("forbidden"))
    print("bad_korean:", check.get("bad_korean"))
    print("missing_sections:", check.get("missing_sections"))
    print("has_end:", check.get("has_end"))
    print("\nfinal_answer_preview:")
    print(result["final_answer"][:700])
    print()
