# RoadSafe Stage Data Plan

## Stage 1. Law DAPT / CLM
목적: 도로교통 법률 언어와 기본 개념 적응  
데이터: 도로교통법, 생활법령 교통·운전, 생활법령 자전거 운전자, 공공기관 해설 자료  
방식: Causal Language Modeling  
목표: 깨끗한 법률·해설 텍스트 1~3MB 이상, 확장 목표 5~20MB

## Stage 2. Accident Structure SFT
목적: 사고 구조, 사실관계, 과실 쟁점, 수정요소 학습  
데이터: 과실비율 인정기준, 판례, 보험 분쟁 사례, 사고 유형별 시나리오  
목표: 500~1000건 이상

## Stage 3. User Instruction SFT
목적: 실제 사용자 질문에 대한 구조화 답변 학습  
데이터: 사용자 질문형 QA, 사고 후 대처 질문, 보험/분쟁 질문, 정보 부족 질문  
목표: 500~1000개 권장, 최소 300~500개

## Stage 4. Hard Case Correction SFT
목적: 평가 중 확인된 오답 유형 교정  
데이터: 실제 오답 기반 hard case QA  
목표: 50~200개, 소수 고품질
