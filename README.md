# RoadSafe LLM FineTuning

도로교통 사고 상황 분석을 위한 법률 도메인 특화 LLM FineTuning 프로젝트입니다.

본 프로젝트는 교통사고 상황 설명을 입력받아 사고 유형, 사실관계, 확인 필요 자료, 관련 근거, 핵심 쟁점, 대처 방법, 주의 문구를 구조화하여 제공하는 RoadSafe AI 시스템 구축을 목표로 합니다.

현재 저장소에는 프로젝트 개발 전 문서인 PRD와 요구사항 분석서를 먼저 정리해두었으며, 이후 Multi-Stage QLoRA 학습 코드, 평가 스크립트, RAG 구성, Safety Filter, Flask REST API 관련 파일을 단계적으로 추가할 예정입니다.

## 프로젝트 개요

* **프로젝트명**: RoadSafe LLM FineTuning
* **주제**: 도로교통 사고 분석을 위한 법률 도메인 특화 LLM 시스템
* **핵심 기술**: Llama 3.2 3B, QLoRA, LoRA, RAG, BM25, FAISS, Safety Filter, Flask REST API
* **프로젝트 기간**: 2026.05.21 ~ 2026.05.31
* **주요 목적**:

  * 교통사고 상황을 구조화된 답변으로 정리
  * 법률·생활법령·과실비율 자료 기반 RAG 검색 적용
  * 과실비율 및 법적 책임 단정 방지
  * 정형 사고와 비정형 사고를 함께 고려한 안전한 상담형 LLM 구축

## 핵심 방향

RoadSafe LLM FineTuning은 단순 질의응답 챗봇이 아니라, 교통사고 도메인의 신뢰성과 안전성을 고려한 LLM 시스템을 목표로 합니다.

특히 다음 원칙을 중요하게 다룹니다.

* 과실비율을 단정하지 않는다.
* 법적 책임을 확정하지 않는다.
* 정보가 부족한 경우 추가 확인 자료를 요청한다.
* 블랙박스, CCTV, 신호, 속도, 충돌 위치 등 증거 자료에 따라 판단이 달라질 수 있음을 안내한다.
* 법률·과실 쟁점은 RAG 검색 근거를 우선 활용한다.
* 위험한 단정 표현은 Safety Filter로 후처리한다.

## 주요 구성 예정

```text
사용자 사고 설명
→ Input Normalizer
→ Emergency Router
→ Hybrid Retriever(BM25 + FAISS)
→ Rule Card / Category Boost
→ Prompt Builder
→ Fine-tuned Llama 3.2 3B
→ Safety Filter
→ Flask REST API Response
```

## Multi-Stage FineTuning 계획

| Stage   | 목적                           |
| ------- | ---------------------------- |
| Stage 1 | 도로교통 법률 언어 적응                |
| Stage 2 | 사고 구조와 과실 쟁점 학습              |
| Stage 3 | 사용자 QA 응답 형식 학습              |
| Stage 4 | Hard Case Correction         |
| Final   | RAG + Safety Filter + API 통합 |

## 프로젝트 문서

프로젝트 개발 전 기획 및 요구사항 문서는 `project_documents/` 폴더에서 확인할 수 있습니다.

```text
project_documents/
├─ RoadSafe_LLM_FineTuning_PRD.pdf
└─ RoadSafe_LLM_FineTuning_Requirements_Analysis.pdf
```

## 저장소 관리 기준

학습 모델, 체크포인트, 대용량 데이터, TensorBoard 로그, 벡터 인덱스 등은 GitHub에 직접 포함하지 않습니다.

GitHub에는 주로 다음 항목을 관리합니다.

* 학습 및 평가 스크립트
* PRD / 요구사항 분석서 / 보고서
* 작은 평가 요약 파일
* README 및 실행 가이드
* RAG/API 구성 코드

제외 대상 예시는 다음과 같습니다.

* 모델 가중치 파일
* 학습 체크포인트
* 대용량 JSONL 데이터
* TensorBoard 로그
* FAISS/BM25 인덱스 파일
* 실행 결과물

## 향후 추가 예정

* Stage3 v3 데이터 생성 스크립트
* Golden Eval 평가 스크립트
* Multi-Stage QLoRA 학습 코드
* RAG 검색 모듈
* Safety Filter
* Flask REST API
* 최종 실험 결과 보고서
* 트러블슈팅 보고서

## 관련 프로젝트

* **OTT ML Web Service**: 1차 ML 기반 웹 서비스 프로젝트
* **RoadSafe Vision**: 2차 YOLO 기반 자전거 헬멧 탐지 프로젝트
* **RoadSafe LLM FineTuning**: 3차 도로교통 사고 분석 LLM 프로젝트
