"""정책 문서 RAG 파이프라인.

data/policy 의 Markdown 문서를 섹션(##) 단위로 나누고,
Chroma 벡터 검색과 BM25 키워드 검색을 섞은 하이브리드 검색기를 만든다.
"""
from functools import lru_cache
from pathlib import Path

from kiwipiepy import Kiwi
from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "data" / "policy"
CHROMA_DIR = ROOT / "chroma_db"
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"
TOP_K = 3

_kiwi = Kiwi()


def korean_tokenizer(text: str) -> list[str]:
    """BM25용 한국어 형태소 토크나이저. 명사·어간·숫자·외래어만 남긴다."""
    keep = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "SL", "SN"}
    return [t.form for t in _kiwi.tokenize(text) if t.tag in keep]


def load_chunks() -> list[Document]:
    """정책 문서를 섹션 단위 청크로 나눈다. 청크마다 doc_id(파일명#섹션)를 붙인다."""
    splitter = MarkdownHeaderTextSplitter([("#", "title"), ("##", "section")], strip_headers=False)
    chunks = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for i, doc in enumerate(splitter.split_text(text)):
            section = doc.metadata.get("section") or "개요"
            doc.metadata.update({
                "doc_id": f"{path.stem}#{section}",
                "source": f"data/policy/{path.name}",
                "title": doc.metadata.get("title", path.stem),
                "chunk": i,
            })
            # 섹션 청크에 문서 제목을 붙여 검색 정확도를 높인다
            if doc.metadata["title"] not in doc.page_content:
                doc.page_content = f"[{doc.metadata['title']}]\n{doc.page_content}"
            chunks.append(doc)
    return chunks


@lru_cache(maxsize=1)
def get_retriever() -> EnsembleRetriever:
    """하이브리드 검색기를 한 번만 만들어 돌려준다. Chroma DB 가 없으면 새로 만든다."""
    chunks = load_chunks()
    embeddings = BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION)
    db = Chroma(collection_name="policy", embedding_function=embeddings, persist_directory=str(CHROMA_DIR))
    if db._collection.count() != len(chunks):
        db.reset_collection()
        db.add_documents(chunks, ids=[f"{c.metadata['doc_id']}:{c.metadata['chunk']}" for c in chunks])
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=korean_tokenizer)
    bm25.k = TOP_K
    vector = db.as_retriever(search_kwargs={"k": TOP_K})
    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.4, 0.6])


def search(query: str) -> list[dict]:
    """질문과 관련된 정책 문서 청크를 [{doc_id, source, text}] 형태로 돌려준다 (최대 3개)."""
    docs = get_retriever().invoke(query)[:TOP_K]
    return [{"doc_id": d.metadata["doc_id"], "source": d.metadata["source"], "text": d.page_content} for d in docs]
