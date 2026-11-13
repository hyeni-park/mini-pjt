# 미니 PJT: 부동산 추천 에이전트

## 무엇을 푸나
수도권 무주택 직장인이 "내 연봉·현금으로 어디까지 살 수 있는지"를 알려면 시세 앱·은행 계산기·정책 사이트를 오가야 한다. 이 에이전트는 연소득·보유 현금·직장 위치·희망 평형을 받아 **대출 한도 → 매수 가능 금액 → 통근 가능 지역 → 예산 안의 실거래 단지**까지 대화 한 번으로 추천하고, 모든 숫자에 출처와 기준일을 붙인다. (기획: [SERVICE.md](SERVICE.md))

## 활용한 패턴 (Day 1~7)
- Day 1: LCEL + Pydantic 구조화 출력 — 질문에서 연소득·현금·직장·평형을 `UserConditions` 로 추출하고 대화별로 누적 (멀티턴)
- Day 2: RAG 하이브리드 검색 — 정책 문서를 섹션 단위로 나눠 Chroma(Titan 임베딩) + BM25(Kiwi 형태소) 앙상블
- Day 3: ReAct — `create_agent` 가 질문 유형에 따라 도구 9개 중 필요한 것을 골라 호출
- Day 4: 도구 다중 — 추천 질문 하나에 대출 계산 → 통근 후보 지역 → 국토부 실거래 API 요약을 이어서 호출
- Day 5: 가드레일 — 투자 단정·불법·범위 밖·프롬프트 인젝션 거절, 숫자는 도구 결과만 사용
- Day 5: HITL — 조건 저장·삭제 전에 `HumanInTheLoopMiddleware` 로 멈추고 사용자 승인
- Day 5: 미들웨어 — `PIIMiddleware` 로 주민번호·전화·계좌번호 마스킹, 사용자 ID 없으면 메모리 도구를 숨기는 게이트
- Day 7: 장기 메모리 — LangGraph `InMemoryStore` 에 사용자가 승인한 계산 조건만 저장하고 새 대화에서 불러오기
- Day 7: Observability — LangSmith 트레이스 + 응답 `trace` 필드
- Day 7: 평가 — LLM-Judge + 규칙 검증 + RAGAS ([1차](evaluation/round1_report.md) · [2차](evaluation/round2_report.md) 리포트)

## 아키텍처
```
POST /query {"question", "session_id"(선택), "user_id"(선택)}
  └─ agent.run()
       ├─ ① 조건 추출: LCEL + Pydantic (Haiku 4.5) → 대화별 누적 조건
       └─ ② create_agent (Sonnet 4.5)
            미들웨어: 메모리 도구 게이트 → PII 마스킹 → HITL 승인
            도구: calc_loan_budget (대출 한도, data/loan_rules.json)
                  get_candidate_regions / region_code_lookup (지역, data/regions.json)
                  summarize_trades / apt_trades (국토부 실거래 API)
                  search_policy (정책 문서 RAG, data/policy/)
                  save / load / delete_conditions (장기 메모리)
  └─ {"answer", "contexts", "trace", "session_id"} → LangSmith 트레이스
```
- 숫자 계산은 모두 도구 안에서 하고, 답변 마지막 두 줄은 `출처:`·`기준일:` 로 고정한다.
- 채팅 화면(`src/static/index.html`)은 같은 서버의 `/` 에서 열리며 스트리밍으로 답한다.

## 실행 방법
```bash
# .env 에 AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION, DATA_GO_KR_API_KEY, LANGSMITH_* 설정
pip install -r requirements.txt
uvicorn server:app --app-dir src --port 8000     # 채팅 화면 http://localhost:8000 · API 문서 /docs

# Docker
docker build -t real-estate-agent .
docker run --env-file .env -p 8000:8000 real-estate-agent

# 평가
pip install ragas==0.4.3 "instructor>=1.11" datasets==5.2.0 && pip install scikit-network --no-deps
python evaluation/run_eval.py
```

## RAGAS 평가 결과
- context_recall: 0.75 (정책 RAG 2건)
- context_precision: 0.92 (정책 RAG 2건)
- faithfulness: 0.82 (근거가 있는 17건)
- answer_relevancy: 0.30 (근거가 있는 17건)
- answer_relevancy 는 답변에서 질문을 거꾸로 만들어 비교하는 방식이라, 금액·단지 목록이 많은 한국어 답변(0.1~0.3)과 "조회 결과 없음" 안내(0.00)에서 낮게 나온다. 질문에 맞게 답했는지는 LLM-Judge(21/21)로 함께 확인했다.

## 인-아웃 세트 통과율 (자체 평가)
- 1차 (Day 9 종료): 21 / 21 통과 — 단, 성공 기준 8개 중 근거 포함률 83% (목표 90%) 미달
- 2차 (Day 10 개선 후): 21 / 21 통과 — 성공 기준 8개 모두 달성 (근거 포함률 100%, 도구 선택률 100%, 평균 12.2초)
- 개선폭: 통과 건수 +0건, 근거 포함률 83% → 100%, 도구 선택률 94% → 100% (주요 개선 사항: 불법 요청 거절 시 근거 없는 처벌 수위 제거, 정책 답변을 검색 문서 내용으로 한정, 지역·직장 위치는 반드시 도구로 확인, 숫자 답변에 출처·기준일이 빠지면 도구 결과로 보완)
- 비교 시 주의: 1·2차 사이에 근거 포함률 집계 규칙(금액·건수·비율만 집계)과 채점 모델(Haiku → Sonnet)이 함께 바뀌어 수치를 그대로 비교할 수 없다. 1차 리포트 분석 기준으로 새 집계 규칙을 적용하면 1차 근거 포함률은 약 94%(15/16)로 추정된다. 2차의 100%는 출처가 빠진 답변에 도구 결과 출처를 코드로 붙이는 후처리를 포함한 값이다.
- 2차 점검 중 추가 수정: 저장 승인 대기 중 '예산…'처럼 승인어로 시작하는 새 질문을 승인으로 오판하던 문제, 실거래 API 오류를 '조회 결과 없음'으로 답하던 문제(이제 '조회 실패'로 구분), 트레이스에 개인정보 원문이 남던 문제, 금액 없는 가정 질문("예산 더 늘리면?")에 예시 숫자를 지어내던 문제

## 트라이앤에러 회고
- 시도했지만 실패한 접근 · 왜 실패했는가
  - 실거래 원본을 LLM에 넘겨 단지를 고르게 하려 했으나, 한 지역·월 거래가 수백 건이라 토큰이 크고 숫자를 틀릴 위험이 있었다 → 도구가 API를 직접 부르고 요약만 반환하도록 바꿨다.
  - 개인정보 안내를 LLM이 쓰게 했더니 "안심하세요" 같은 어색한 문장이 나오고, 스트리밍 중 도구 호출 때 지워져 답변에서 빠졌다 → 마스킹이 일어나면 코드가 고정 안내 문구를 붙이도록 바꿨다.
  - 평가 채점기를 Haiku로 두었더니 2026년 실거래 숫자를 "미래라서 지어낸 숫자"로 판단하고, 멀티턴은 2턴만 보고 "이전 조건을 안 이어받았다"고 오판했다 → Sonnet으로 바꾸고 오늘 날짜·도구 결과·도구 입력·이전 턴 답변을 근거로 함께 넘겼다.
  - 사용자 ID를 입력칸으로 받았더니 비워 둔 채 저장을 요청하면 "기능을 쓸 수 없다"는 모호한 답이 나왔다 → 브라우저가 익명 ID를 자동 생성하고, ID가 없으면 메모리 도구 자체를 모델에서 숨겼다.
- 최종 채택한 접근 · 왜 그것으로 갔는가
  - 단일 ReAct 에이전트 + 미들웨어: 도구가 9개이고 흐름이 대출 → 지역 → 실거래로 순차적이라, Supervisor로 나누면 LLM 호출만 2~3배 늘어 응답 시간(목표 30초)과 비용이 커진다. MCP도 도구를 외부에 공유할 필요가 없어 쓰지 않았다.
  - 조건 추출을 Pydantic 체인으로 분리: "현금 5천 더"는 증감 필드로 받아 코드가 더해, 멀티턴 재계산에서 LLM이 숫자를 계산하지 않게 했다.
  - 저장·삭제에만 HITL: 조회·계산은 즉시 답하고, 되돌리기 어려운 개인정보 저장·삭제만 승인을 받아 평가 흐름을 끊지 않았다.
- 남은 한계 · 향후 개선 방향
  - LTV·DSR·금리는 공개 정리 자료 기반 가정값(원문 확인 전)이고 기존 부채·부대비용과 15억 초과 주택의 총액 한도 차등(4억·2억)은 반영하지 않는다 → 금융위 원문 확인, 부채 입력·가격 구간별 한도 추가
  - 직장 거점 3곳·후보 지역 12곳 정적 테이블 → 대중교통 길찾기 API로 통근 시간 계산
  - 메모리는 서버 메모리(재시작 시 사라짐)이고 인증이 없다 → 로그인 계정 ID와 DB 저장소
  - 추천 질문은 15~25초 걸린다 → 답변 길이 축소, 대출 계산·후보 지역 병렬 호출

## 핵심 코드 위치
- `src/agent.py:200` — 메인 에이전트 그래프 (`create_agent` + 미들웨어 + 단기·장기 메모리)
- `src/agent.py:86` — 조건 추출 LCEL 체인 (Pydantic `UserConditions`)
- `src/agent.py:51`, `:180`, `:192` — PII 미들웨어, 메모리 도구 게이트, HITL 미들웨어
- `src/agent.py:366` — 질문 처리 흐름 (승인 대기 처리), `:319` 응답 조립 (출처·기준일 보완)
- `src/tools.py` — 도메인 도구 9개 (`:128` 대출 계산식, `:211` 실거래 요약)
- `src/retriever.py:55` — RAG 파이프라인 (Chroma + BM25 하이브리드)
- `src/server.py:55` — FastAPI `POST /query`
- `evaluation/run_eval.py` — 평가 스크립트 (LLM-Judge·규칙 검증·RAGAS)
