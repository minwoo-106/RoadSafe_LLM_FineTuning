from pathlib import Path
import sys

# scripts 폴더 안의 controlled RAG 함수를 재사용
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.append(str(SCRIPT_DIR))

from answer_with_retrieval_v3_controlled import (
    retrieve_rule_cards,
    retrieve_chunks,
    build_answer,
)


def answer_question(question: str, top_k_rules: int = 3, top_k_chunks: int = 4) -> dict:
    """
    RoadSafe Controlled RAG answer engine.

    Returns:
        {
            "question": str,
            "rule_cards": list,
            "chunks": list,
            "answer": str
        }
    """
    rule_cards = retrieve_rule_cards(question, top_k=top_k_rules)
    chunks = retrieve_chunks(question, top_k=top_k_chunks)
    answer = build_answer(question, rule_cards, chunks)

    return {
        "question": question,
        "rule_cards": rule_cards,
        "chunks": chunks,
        "answer": answer,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top_k_rules", type=int, default=3)
    parser.add_argument("--top_k_chunks", type=int, default=4)
    args = parser.parse_args()

    result = answer_question(
        args.question,
        top_k_rules=args.top_k_rules,
        top_k_chunks=args.top_k_chunks,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))
