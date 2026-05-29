import json
import math
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

CHUNK_PATH = Path("data/processed/roadsafe_law_chunks_v1.jsonl")

def tokenize(text: str):
    text = text.lower()
    words = re.findall(r"[가-힣A-Za-z0-9]+", text)

    tokens = []
    for w in words:
        tokens.append(w)
        # 한국어 복합어 대응용 2~3글자 ngram
        if re.search(r"[가-힣]", w) and len(w) >= 3:
            for n in (2, 3):
                for i in range(len(w) - n + 1):
                    tokens.append(w[i:i+n])
    return tokens

def load_chunks():
    rows = []
    with CHUNK_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

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

def score_bm25(query, rows, doc_tokens, df, k1=1.5, b=0.75):
    q_tokens = tokenize(query)
    q_count = Counter(q_tokens)

    N = len(rows)
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
            idf = math.log(1 + (N - term_df + 0.5) / (term_df + 0.5))
            tf = c[term]
            denom = tf + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf * (k1 + 1) / denom)

        if score > 0:
            scores.append((score, idx))

    scores.sort(reverse=True)
    return scores

def main():
    if len(sys.argv) < 2:
        print('usage: python scripts/search_law_chunks.py "검색어" [top_k]')
        raise SystemExit(1)

    query = sys.argv[1]
    top_k = int(sys.argv[2]) if len(sys.argv) >= 3 else 5

    rows = load_chunks()
    doc_tokens, df = build_index(rows)
    scores = score_bm25(query, rows, doc_tokens, df)

    print(f"[QUERY] {query}")
    print(f"[RESULTS] top {top_k}")

    for rank, (score, idx) in enumerate(scores[:top_k], 1):
        row = rows[idx]
        text = re.sub(r"\s+", " ", row["text"]).strip()
        print("\n" + "=" * 80)
        print(f"[{rank}] score={score:.3f}")
        print(f"chunk_id: {row['chunk_id']}")
        print(f"source: {row['source_file']}")
        print(f"category: {row['category']}")
        print("-" * 80)
        print(text[:900])

if __name__ == "__main__":
    main()
