# 프로젝트 규칙

## 기술 스택
- Python, LangChain, LangGraph
- 모델은 Amazon Bedrock (ChatBedrockConverse, region `us-east-1`)
  - 에이전트: `global.anthropic.claude-sonnet-4-5-20250929-v1:0`
  - 조건 추출: `global.anthropic.claude-haiku-4-5-20251001-v1:0`
  - 평가(LLM-Judge·RAGAS): 에이전트와 같은 Sonnet 4.5
- Agent 생성은 langchain.agents 의 create_agent 를 쓴다
- 사용자 조건(연소득·보유 현금·직장 위치·희망 평형) 추출은 LCEL 체인과 Pydantic 구조화 출력(with_structured_output)으로 한다
- 트레이스는 LangSmith 로 기록한다 (LANGSMITH_* 값은 .env 에서 읽는다)

## 폴더 구조 (제출 규약. 바꾸지 않는다)
src/agent.py       메인 에이전트 그래프
src/tools.py       도메인 도구
src/retriever.py   RAG 파이프라인
src/server.py      FastAPI 서버 (POST /query)
src/static/index.html  시연용 채팅 화면 (서버가 / 에서 띄움)
data/              사용한 문서와 데이터
evaluation/        test_queries.csv, run_eval.py(평가 스크립트), round1_report.md·round2_report.md(1·2차 평가 리포트)
requirements.txt   실행 패키지 (평가 패키지 설치는 README 참고)
Dockerfile         서버 이미지

## 주고받는 형식 (제출 규약)
- POST /query 로 받고 question 필드를 읽는다
- session_id 는 선택 필드다. 있으면 같은 대화로 이어서 답하고, 없으면 새 대화로 처리한다
- user_id 는 선택 필드다. 있으면 계산 조건 저장·불러오기(장기 메모리)를 쓰고, 없으면 그 기능만 꺼진다
- 답은 answer, contexts, trace 세 키로 돌려준다
  - contexts: [{"doc_id": "...", "text": "..."}] 형태. 검색 청크와 도구 출력을 담는다 (도구 출력의 doc_id 는 도구 이름)
  - trace: [{"step": "...", "input": "...", "output": "..."}] 형태. step 에는 호출한 도구 이름을 쓴다

## 도메인 규칙
- 서비스 동작(추천 기준, 가드레일)은 SERVICE.md 를 따른다
- 도구 이름은 SERVICE.md 도구 표와 똑같이 쓴다
- 금액은 만 원 단위로 다룬다
- 숫자 계산은 도구 안에서만 한다. 대출 기준 숫자는 data/loan_rules.json 에서만 읽는다
- summarize_trades 는 실거래 API 를 직접 호출(numOfRows=1000)하고 요약만 반환한다. 거래 원본을 LLM 에 넘기지 않는다
- 숫자가 들어간 답변은 마지막 두 줄을 `출처: ...`, `기준일: YYYY-MM-DD` 로 끝낸다

## 코드 규칙
- 파일 하나에 한 가지 역할만 둔다
- 함수와 도구에는 한국어 docstring 을 쓴다
- 비밀 값은 .env 에서 읽고 코드에 적지 않는다

## 하지 말 것
- 요청하지 않은 파일을 새로 만들지 않는다
- 기존 파일을 통째로 다시 쓰지 않는다. 바뀐 부분만 고친다