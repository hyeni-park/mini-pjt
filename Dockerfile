# 부동산 추천 에이전트 서버 이미지
#   빌드: docker build -t real-estate-agent .
#   실행: docker run --env-file .env -p 8000:8000 real-estate-agent
#   접속: http://localhost:8000 (채팅 화면), http://localhost:8000/docs (API 문서)
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY data ./data

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# 정책 문서 벡터 DB(chroma_db)는 첫 정책 질문 때 컨테이너 안에서 자동으로 만든다
CMD ["uvicorn", "server:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
