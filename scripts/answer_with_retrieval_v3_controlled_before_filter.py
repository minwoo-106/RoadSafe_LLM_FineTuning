import argparse
import json
import math
import re
from pathlib import Path
from collections import Counter, defaultdict


CHUNK_PATH = Path("data/processed/roadsafe_law_chunks_v1.jsonl")
RULE_CARD_PATH = Path("data/processed/roadsafe_rule_cards_v1.jsonl")
OUT_DIR = Path("outputs/rag_answers_v3_controlled")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CAUTION = (
    "이 답변은 일반적인 교통사고·도로교통 쟁점 안내입니다. "
    "실제 과실비율과 법적 책임은 신호, 속도, 충돌 위치, 블랙박스, CCTV, 진술, "
    "보험사·수사기관·법원 판단에 따라 달라질 수 있습니다. "
    "따라서 특정 비율로 단정하지 말아야 합니다."
)


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


def score_rule_card(query, card):
    expanded = expand_query(query)
    q_tokens = set(tokenize(expanded))

    score = 0

    for kw in card.get("keywords", []):
        if kw in expanded:
            score += 8

        for t in tokenize(kw):
            if t in q_tokens:
                score += 1

    for t in tokenize(card.get("title", "")):
        if t in q_tokens:
            score += 1

    # 강제 보정
    if "자전거횡단도" in query and "자전거횡단도" in card.get("keywords", []):
        score += 40

    if ("전동킥보드" in query or "킥보드" in query or "PM" in query) and "횡단보도" in query:
        if "전동킥보드" in card.get("keywords", []) or "PM" in card.get("keywords", []):
            score += 40

    if "우회전" in query and "보행자" in query:
        if "보행자" in card.get("keywords", []) and "우회전" in card.get("keywords", []):
            score += 40

    if "고속도로" in query and "1차로" in query:
        if "고속도로" in card.get("keywords", []):
            score += 40

    if "급제동" in query or "급브레이크" in query:
        if "급제동" in card.get("keywords", []) or "안전거리" in card.get("keywords", []):
            score += 40

    if "신호등 없는" in query or "신호 없는" in query:
        if "신호등 없는 교차로" in card.get("keywords", []) or "교통정리 없는 교차로" in card.get("keywords", []):
            score += 40

    return score


def retrieve_rule_cards(query, top_k=3):
    cards = load_jsonl(RULE_CARD_PATH)
    scored = []

    for card in cards:
        score = score_rule_card(query, card)
        if score > 0:
            scored.append((score, card))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, card in scored[:top_k]:
        item = card.copy()
        item["score"] = score
        results.append(item)

    return results


def build_chunk_index(rows):
    doc_tokens = []
    df = defaultdict(int)

    for row in rows:
        c = Counter(tokenize(row["text"]))
        doc_tokens.append(c)
        for t in c:
            df[t] += 1

    return doc_tokens, df


def bm25(query, rows, doc_tokens, df, k1=1.5, b=0.75):
    q_count = Counter(tokenize(expand_query(query)))
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
    rows = load_jsonl(CHUNK_PATH)
    doc_tokens, df = build_chunk_index(rows)
    scores = bm25(query, rows, doc_tokens, df)

    results = []
    for score, idx in scores[:top_k]:
        item = rows[idx].copy()
        item["score"] = score
        results.append(item)

    return results


def infer_facts(question):
    q = question

    facts = {
        "자동차": "확인 필요",
        "상대 교통주체": "확인 필요",
        "충돌 지점": "확인 필요",
        "확인 필요": [],
    }

    if "전동킥보드" in q or "킥보드" in q or "PM" in q or "개인형" in q:
        facts["자동차"] = "횡단보도 또는 교차 지점을 진행하던 자동차"
        facts["상대 교통주체"] = "전동킥보드/PM 운전자"
        facts["충돌 지점"] = "횡단보도 또는 자전거횡단도 부근"
        facts["확인 필요"] = [
            "PM을 탄 채 이동했는지, 내려서 끌고 보행했는지",
            "일반 횡단보도인지 자전거횡단도가 함께 있는지",
            "보행자 신호와 차량 신호",
            "자동차의 전방주시·감속·회피 가능성",
        ]

    elif "자전거횡단도" in q:
        facts["자동차"] = "자전거횡단도 부근을 지나가던 자동차"
        facts["상대 교통주체"] = "자전거횡단도를 통행 중인 자전거 또는 자전거등"
        facts["충돌 지점"] = "자전거횡단도 또는 그 앞 정지선 부근"
        facts["확인 필요"] = [
            "사고 지점이 실제 자전거횡단도인지",
            "자전거 신호 또는 보행자 신호 상태",
            "자동차가 자전거횡단도 앞에서 일시정지했는지",
            "자전거의 갑작스러운 진입이나 신호위반 여부",
        ]

    elif "고속도로" in q:
        facts["자동차"] = "고속도로 1차로를 주행 중이던 앞차와 뒤차"
        facts["상대 교통주체"] = "경적, 앞지르기, 급제동을 한 뒤차"
        facts["충돌 지점"] = "고속도로 1차로 또는 앞지르기·진로변경 이후 지점"
        facts["확인 필요"] = [
            "앞차의 1차로 지속 주행이 지정차로 위반 쟁점인지",
            "뒤차의 경적·근접주행·앞지르기 과정",
            "뒤차의 급제동에 정당한 사유가 있었는지",
            "블랙박스상 차간거리와 제동 시점",
        ]

    elif "급제동" in q or "급브레이크" in q:
        facts["자동차"] = "앞차와 뒤차"
        facts["상대 교통주체"] = "급제동한 앞차 또는 추돌한 뒤차"
        facts["충돌 지점"] = "같은 방향 진행 중 후방추돌 지점"
        facts["확인 필요"] = [
            "뒤차의 안전거리 확보 여부",
            "앞차 급제동의 정당한 사유",
            "보복성 또는 위협성 급제동인지",
            "제동 전후 도로 상황과 주변 장애물",
        ]

    elif "신호등 없는" in q or "신호 없는" in q:
        facts["자동차"] = "신호등 없는 교차로에 진입한 차량들"
        facts["상대 교통주체"] = "직진 차량, 좌회전 차량 또는 교차 방향 차량"
        facts["충돌 지점"] = "교통정리 없는 교차로 내부 또는 진입부"
        facts["확인 필요"] = [
            "선진입 여부",
            "대로·소로 또는 도로 폭 차이",
            "우측도로 차량 여부",
            "직진·좌회전 관계와 양측 서행 여부",
        ]

    elif "우회전" in q and "보행자" in q:
        facts["자동차"] = "우회전하던 자동차"
        facts["상대 교통주체"] = "횡단보도를 통행 중이거나 통행하려던 보행자"
        facts["충돌 지점"] = "횡단보도 또는 우회전 경로와 횡단보도가 만나는 지점"
        facts["확인 필요"] = [
            "보행자 신호와 차량 신호",
            "자동차의 일시정지 또는 서행 여부",
            "보행자의 횡단 시작 시점",
            "운전자의 전방·측방 확인 여부",
        ]

    else:
        facts["확인 필요"] = [
            "사고 위치",
            "각 교통주체의 진행 방향",
            "신호와 속도",
            "충돌 위치와 블랙박스 여부",
        ]

    return facts


def build_issues(question, rule_cards):
    issues = []

    for card in rule_cards:
        hint = card.get("answer_hint", "")
        parts = [x.strip() for x in re.split(r"[.。]\s*", hint) if x.strip()]
        for p in parts:
            if p not in issues:
                issues.append(p)

    # 질문별 필수 쟁점 강제 삽입
    if ("전동킥보드" in question or "킥보드" in question or "PM" in question) and "횡단보도" in question:
        forced = [
            "PM을 탄 채 일반 횡단보도를 건넜는지, 내려서 끌고 보행했는지 구분해야 합니다",
            "자전거횡단도 표시가 함께 있었는지 확인해야 합니다",
        ]
        for x in forced:
            if x not in issues:
                issues.insert(0, x)

    if "자전거횡단도" in question:
        forced = [
            "자동차가 자전거횡단도 앞에서 일시정지했는지 확인해야 합니다",
            "자전거등의 횡단을 방해하거나 위험하게 했는지가 핵심입니다",
        ]
        for x in forced:
            if x not in issues:
                issues.insert(0, x)

    if "고속도로" in question and "1차로" in question:
        forced = [
            "고속도로 1차로 정속주행 쟁점과 뒤차의 위협적 운전 쟁점은 분리해서 봐야 합니다",
            "뒤차의 경적, 근접주행, 앞지르기 후 급제동이 난폭운전 또는 보복운전 정황인지 확인해야 합니다",
        ]
        for x in forced:
            if x not in issues:
                issues.insert(0, x)

    return issues[:6]


def build_answer(question, rule_cards, chunks):
    facts = infer_facts(question)
    issues = build_issues(question, rule_cards)

    lines = []

    lines.append("[상황 요약]")
    lines.append(question)
    lines.append("")

    lines.append("[사실관계 정리]")
    lines.append(f"- 자동차: {facts['자동차']}")
    lines.append(f"- 상대 교통주체: {facts['상대 교통주체']}")
    lines.append(f"- 충돌 지점: {facts['충돌 지점']}")
    lines.append("- 확인 필요:")
    for x in facts["확인 필요"]:
        lines.append(f"  - {x}")
    lines.append("")

    lines.append("[관련 근거]")
    for card in rule_cards[:2]:
        lines.append(f"- 규칙 카드({card['rule_id']} / {card['source_hint']}): {card['rule_summary']}")
    if chunks:
        chunk_refs = ", ".join([f"{c['chunk_id']}({c['category']})" for c in chunks[:3]])
        lines.append(f"- 원문 검색 근거: {chunk_refs}")
    lines.append("")

    lines.append("[핵심 쟁점]")
    for x in issues:
        lines.append(f"- {x}")
    lines.append("")

    lines.append("[대처 방법]")
    actions = [
        "부상 여부와 2차 사고 위험을 먼저 확인합니다.",
        "블랙박스 원본과 현장 사진을 보존합니다.",
        "신호, 속도, 충돌 위치, 진행 방향을 시간순으로 정리합니다.",
        "검색된 근거와 실제 사고 상황이 일치하는지 확인합니다.",
        "보험사에 접수하고 필요하면 경찰 신고 또는 전문가 상담을 검토합니다.",
    ]
    for i, x in enumerate(actions, 1):
        lines.append(f"{i}. {x}")
    lines.append("")

    lines.append("[주의]")
    lines.append(CAUTION)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", type=str)
    parser.add_argument("--top_k_rules", type=int, default=3)
    parser.add_argument("--top_k_chunks", type=int, default=4)
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

    answer = build_answer(args.question, rule_cards, chunks)

    print("\n" + "=" * 80)
    print("[CONTROLLED ANSWER]")
    print(answer)

    if args.save:
        safe_name = re.sub(r"[^가-힣A-Za-z0-9]+", "_", args.question)[:60].strip("_")
        out_path = OUT_DIR / f"{safe_name}.json"

        payload = {
            "question": args.question,
            "rule_cards": rule_cards,
            "chunks": chunks,
            "answer": answer,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print("\n[OK] saved:", out_path)


if __name__ == "__main__":
    main()
