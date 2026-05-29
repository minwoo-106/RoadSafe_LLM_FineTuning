import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel


MODEL_ID = "meta-llama/Llama-3.2-3B"
ADAPTER_DIR = Path("outputs/final_candidates/roadsafe_stage1_best_adapter")

CHUNK_PATH = Path("data/processed/roadsafe_law_chunks_v1.jsonl")
RULE_CARD_PATH = Path("data/processed/roadsafe_rule_cards_v1.jsonl")

OUT_DIR = Path("outputs/rag_answers_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tokenize(text: str):
    text = text.lower()
    words = re.findall(r"[가-힣A-Za-z0-9]+", text)

    tokens = []
    for w in words:
        tokens.append(w)
        if re.search(r"[가-힣]", w) and len(w) >= 3:
            for n in (2, 3):
                for i in range(len(w) - n + 1):
                    tokens.append(w[i:i+n])
    return tokens


def load_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_chunks():
    return load_jsonl(CHUNK_PATH)


def load_rule_cards():
    return load_jsonl(RULE_CARD_PATH)


def build_index(rows):
    doc_tokens = []
    df = defaultdict(int)

    for row in rows:
        toks = tokenize(row["text"])
        c = Counter(toks)
        doc_tokens.append(c)
        for t in c:
            df[t] += 1

    return doc_tokens, df


def expand_query(query: str) -> str:
    q = query
    extra = []

    if "자전거횡단도" in q:
        extra.append("자전거횡단도 자전거등 횡단 방해 위험 일시정지 도로교통법 제15조의2")

    if "횡단보도" in q and ("전동킥보드" in q or "킥보드" in q or "PM" in q or "개인형" in q):
        extra.append("개인형 이동장치 PM 자전거등 횡단보도 내려서 끌거나 들고 보행 도로교통법 제13조의2 제27조")

    if "우회전" in q and "보행자" in q:
        extra.append("우회전 횡단보도 보행자 보호 일시정지 도로교통법 제27조")

    if "우회전" in q and "자전거" in q:
        extra.append("우회전 자전거도로 자전거횡단도 자전거등 횡단 방해 일시정지 진로변경 안전거리")

    if "고속도로" in q and ("1차로" in q or "앞지르기" in q):
        extra.append("고속도로 1차로 앞지르기 차로 지정차로 도로교통법 제60조")

    if "급제동" in q or "급브레이크" in q or "보복" in q:
        extra.append("안전거리 확보 급제동 금지 위험방지 부득이한 경우 난폭운전 도로교통법 제19조 제46조의3")

    if "신호등 없는" in q or "신호 없는" in q:
        extra.append("교통정리 없는 교차로 선진입 도로 폭 우측도로 좌회전 직진 도로교통법 제26조")

    return q + " " + " ".join(extra)


def bm25(query, rows, doc_tokens, df, k1=1.5, b=0.75):
    q_tokens = tokenize(expand_query(query))
    q_count = Counter(q_tokens)

    n_docs = len(rows)
    lengths = [sum(c.values()) for c in doc_tokens]
    avgdl = sum(lengths) / max(1, len(lengths))

    scores = []

    for idx, c in enumerate(doc_tokens):
        dl = lengths[idx]
        score = 0.0

        for term in q_count:
            if term not in c:
                continue

            term_df = df.get(term, 0)
            idf = math.log(1 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            tf = c[term]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1) / denom)

        if score > 0:
            scores.append((score, idx))

    scores.sort(reverse=True)
    return scores


def retrieve_chunks(query, top_k=4):
    rows = load_chunks()
    doc_tokens, df = build_index(rows)
    scores = bm25(query, rows, doc_tokens, df)

    results = []
    for score, idx in scores[:top_k]:
        item = rows[idx].copy()
        item["score"] = score
        results.append(item)

    return results


def score_rule_card(query, card):
    q = expand_query(query)
    q_tokens = set(tokenize(q))

    score = 0

    for kw in card.get("keywords", []):
        if kw in q:
            score += 5

        kw_tokens = tokenize(kw)
        for t in kw_tokens:
            if t in q_tokens:
                score += 1

    title = card.get("title", "")
    for t in tokenize(title):
        if t in q_tokens:
            score += 1

    # 질문별 강제 보정
    if "자전거횡단도" in query and "자전거횡단도" in card.get("keywords", []):
        score += 20

    if ("전동킥보드" in query or "킥보드" in query or "PM" in query) and "횡단보도" in query:
        if "전동킥보드" in card.get("keywords", []) or "PM" in card.get("keywords", []):
            score += 20

    if "고속도로" in query and "1차로" in query:
        if "고속도로" in card.get("keywords", []):
            score += 20

    if "급제동" in query or "급브레이크" in query:
        if "급제동" in card.get("keywords", []) or "안전거리" in card.get("keywords", []):
            score += 20

    if "신호등 없는" in query or "신호 없는" in query:
        if "신호등 없는 교차로" in card.get("keywords", []) or "교통정리 없는 교차로" in card.get("keywords", []):
            score += 20

    return score


def retrieve_rule_cards(query, top_k=3):
    cards = load_rule_cards()

    scored = []
    for card in cards:
        score = score_rule_card(query, card)
        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [{"score": score, **card} for score, card in scored[:top_k]]


def clean_text(text: str, max_chars=900):
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def build_rule_context(cards):
    parts = []

    for i, card in enumerate(cards, 1):
        parts.append(
            f"[규칙 카드 {i}]\n"
            f"- rule_id: {card['rule_id']}\n"
            f"- title: {card['title']}\n"
            f"- source_hint: {card['source_hint']}\n"
            f"- rule_summary: {card['rule_summary']}\n"
            f"- answer_hint: {card['answer_hint']}"
        )

    return "\n\n".join(parts)


def build_chunk_context(chunks):
    parts = []

    for i, ch in enumerate(chunks, 1):
        parts.append(
            f"[원문 근거 {i}]\n"
            f"- chunk_id: {ch['chunk_id']}\n"
            f"- source: {ch['source_file']}\n"
            f"- category: {ch['category']}\n"
            f"- text: {clean_text(ch['text'])}"
        )

    return "\n\n".join(parts)


def load_model():
    print("[LOAD] tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("[LOAD] base model 4bit")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
    )

    print("[LOAD] adapter")
    model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
    model.eval()

    return tokenizer, model


def generate_answer(question, rule_cards, chunks, tokenizer, model, max_new_tokens=760):
    rule_context = build_rule_context(rule_cards)
    chunk_context = build_chunk_context(chunks)

    prompt = f"""너는 RoadSafe AI다.
사용자의 교통사고·도로교통 질문에 대해 아래 [핵심 규칙 카드]를 최우선으로 참고하고, [원문 검색 근거]를 보조 근거로 사용해 답변하라.

작성 규칙:
- [핵심 규칙 카드]의 rule_summary와 answer_hint를 반드시 반영한다.
- [원문 검색 근거]는 보조 근거로만 사용한다.
- 입력 상황에 없는 사실을 만들지 말 것.
- 자동차, 자전거, PM, 보행자의 역할과 진행 방향을 바꾸지 말 것.
- 과실비율을 특정 숫자로 확정하지 말 것.
- 검색 근거에 없는 사실은 "확인 필요"라고 쓸 것.
- 답변은 반드시 아래 섹션 형식을 따른다.

[핵심 규칙 카드]
{rule_context}

[원문 검색 근거]
{chunk_context}

[사용자 질문]
{question}

[답변 형식]
[상황 요약]
...

[사실관계 정리]
- 자동차:
- 상대 교통주체:
- 충돌 지점:
- 확인 필요:

[관련 근거]
- 규칙 카드:
- 원문 근거:

[핵심 쟁점]
- ...

[대처 방법]
1. ...

[주의]
...

답변:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.08,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()

    return answer, rule_context, chunk_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top_k_chunks", type=int, default=4)
    parser.add_argument("--top_k_rules", type=int, default=3)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    print("[QUESTION]", args.question)

    rule_cards = retrieve_rule_cards(args.question, top_k=args.top_k_rules)
    chunks = retrieve_chunks(args.question, top_k=args.top_k_chunks)

    print("\n[RULE CARDS]")
    for i, card in enumerate(rule_cards, 1):
        print(f"{i}. score={card['score']} | {card['rule_id']} | {card['title']}")

    print("\n[RETRIEVED CHUNKS]")
    for i, ch in enumerate(chunks, 1):
        print(f"{i}. score={ch['score']:.3f} | {ch['chunk_id']} | {ch['category']} | {ch['source_file']}")

    tokenizer, model = load_model()
    answer, rule_context, chunk_context = generate_answer(args.question, rule_cards, chunks, tokenizer, model)

    print("\n" + "=" * 80)
    print("[ANSWER]")
    print(answer)

    if args.save:
        safe_name = re.sub(r"[^가-힣A-Za-z0-9]+", "_", args.question)[:60].strip("_")
        out_path = OUT_DIR / f"{safe_name}.json"

        payload = {
            "question": args.question,
            "rule_cards": rule_cards,
            "chunks": chunks,
            "rule_context": rule_context,
            "chunk_context": chunk_context,
            "answer": answer,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print("\n[OK] saved:", out_path)


if __name__ == "__main__":
    main()
