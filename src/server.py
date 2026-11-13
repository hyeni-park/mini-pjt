"""부동산 추천 에이전트 FastAPI 서버.

실행: mini-pjt 폴더에서  uvicorn server:app --app-dir src --port 8000
POST /query {"question": "...", "session_id": "(선택)", "user_id": "(선택)"} → {"answer", "contexts", "trace", "session_id"}
"""
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent import run, run_stream

app = FastAPI(title="부동산 추천 에이전트")
CHAT_PAGE = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def chat_page() -> FileResponse:
    """시연용 채팅 화면을 돌려준다."""
    return FileResponse(CHAT_PAGE)


class QueryRequest(BaseModel):
    """질문 요청. session_id 가 있으면 같은 대화로 이어서 답하고, user_id 가 있으면 조건 저장·불러오기를 쓴다."""
    question: str
    session_id: str | None = None
    user_id: str | None = None


class Context(BaseModel):
    """답변 근거 (검색 청크 또는 도구 출력)."""
    doc_id: str
    text: str


class TraceStep(BaseModel):
    """도구 호출 한 번의 기록."""
    step: str
    input: str
    output: str


class QueryResponse(BaseModel):
    """질문 응답."""
    answer: str
    contexts: list[Context]
    trace: list[TraceStep]
    session_id: str


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> dict:
    """질문을 받아 에이전트 답변과 근거(contexts), 도구 호출 기록(trace)을 돌려준다."""
    return await run_in_threadpool(run, req.question, req.session_id, req.user_id)


@app.post("/query/stream", include_in_schema=False)
def query_stream(req: QueryRequest) -> StreamingResponse:
    """채팅 화면용 스트리밍 응답 (SSE). 글자·도구 호출 이벤트를 보내고, 마지막에 /query 와 같은 응답을 보낸다."""
    def events():
        try:
            for event in run_stream(req.question, req.session_id, req.user_id):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except Exception as exc:  # 화면에 오류를 보여주기 위해 이벤트로 넘긴다
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"
    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict:
    """서버 동작 확인용."""
    return {"status": "ok"}
