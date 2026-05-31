import sys
import re
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "final_route" / "scripts"
sys.path.append(str(SCRIPTS_DIR))

from run_final_roadsafe_inference import load_model, generate_answer  # noqa: E402


app = FastAPI(
    title="RoadSafe LLM API",
    version="1.0.0",
    description="RoadSafe traffic accident consultation LLM API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def make_public_answer(answer: str) -> str:
    """사용자 화면에 보여줄 최종 답변에서 개발/내부용 문장을 제거한다."""
    if not answer:
        return ""

    text = answer.replace("<END>", "").strip()

    # 사용자 화면에서는 RAG 미구현/내부 보강 문장이 섞인 [관련 근거] 섹션을 숨긴다.
    text = re.sub(
        r"\n?\[관련 근거\]\n.*?(?=\n\[핵심 쟁점\]|\n\[대처 방법\]|\n\[주의\]|$)",
        "\n",
        text,
        flags=re.DOTALL,
    )

    # 빈 줄 정리
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

_tokenizer = None
_model = None


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    safety_check: Dict[str, Any]
    raw_answer: Optional[str] = None


def get_loaded_model():
    global _tokenizer, _model

    if _tokenizer is None or _model is None:
        print("[RoadSafe API] loading model...")
        _tokenizer, _model = load_model()
        print("[RoadSafe API] model loaded.")

    return _tokenizer, _model


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "roadsafe-llm-api",
        "model_loaded": _model is not None,
    }


@app.post("/roadsafe/chat", response_model=ChatResponse)
def roadsafe_chat(req: ChatRequest):
    question = req.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="question is required")

    tokenizer, model = get_loaded_model()

    try:
        result = generate_answer(tokenizer, model, question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"inference failed: {e}")

    return {
        "answer": make_public_answer(result.get("final_answer", "")),
        "raw_answer": result.get("raw_answer", ""),
        "safety_check": result.get("safety_check", {}),
    }



# 기존 Windows Flask llm.html 호환용 엔드포인트
@app.post("/api/roadsafe/ask")
def roadsafe_legacy_ask(req: ChatRequest):
    question = req.question.strip()

    if not question:
        return {
            "ok": False,
            "message": "질문을 입력해 주세요.",
        }

    tokenizer, model = get_loaded_model()

    try:
        result = generate_answer(tokenizer, model, question)
    except Exception as e:
        return {
            "ok": False,
            "message": f"RoadSafe LLM 추론 실패: {e}",
        }

    safety_check = result.get("safety_check", {})

    return {
        "ok": True,
        "result": {
            "answer": make_public_answer(result.get("final_answer", "")),
            "raw_answer": result.get("raw_answer", ""),
            "safety_check": safety_check,
            "rule_cards": [],
            "chunks": [],
            "mode": "finetuned_llm_guardrail",
            "forced_fallback_reason": safety_check.get("forced_fallback_reason", ""),
            "fallback_used": safety_check.get("fallback_used", False),
        },
    }



@app.options("/api/roadsafe/ask")
def roadsafe_legacy_ask_options():
    return {"ok": True}
