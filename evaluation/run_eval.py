"""부동산 추천 에이전트 평가 스크립트.

test_queries.csv 의 21건을 에이전트에 보내고 SERVICE.md 5장 성공 기준대로 채점한 뒤,
RAGAS 지표까지 계산해 리포트로 저장한다 (기본 evaluation/round2_report.md, --report 로 파일 이름 지정).

실행 (mini-pjt 폴더에서):
    python evaluation/run_eval.py                 # 에이전트를 직접 불러 실행
    python evaluation/run_eval.py --api http://localhost:8000   # 실행 중인 서버의 POST /query 로 실행
    python evaluation/run_eval.py --fresh         # 저장된 응답을 버리고 처음부터 다시 실행

응답은 evaluation/eval_answers.jsonl 에 한 건씩 저장되므로, 중간에 멈춰도 다시 실행하면 이어서 돈다.
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.request
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from langchain_aws import ChatBedrockConverse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from tools import compute_loan_budget, format_manwon  # noqa: E402

EVAL_DIR = ROOT / "evaluation"
QUERIES = EVAL_DIR / "test_queries.csv"
ANSWERS = EVAL_DIR / "eval_answers.jsonl"
RESULTS = EVAL_DIR / "eval_results.csv"
REPORT = EVAL_DIR / "round2_report.md"   # --report 로 바꿀 수 있다 (1차는 round1_report.md)

JUDGE_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"   # LLM-Judge·RAGAS 평가 모델
EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
REGION = "us-east-1"

# 대출 계산 케이스: id → (연소득, 보유 현금, 답변에 있어야 할 금액 항목)
LOAN_CASES = {
    7: (7000, 20000, "대출_한도_만원"),
    8: (5000, 15000, "매수_가능_금액_만원"),
    15: (15000, 10000, "매수_가능_금액_만원"),
    16: (15000, 30000, "대출_한도_만원"),
}
PII_CASES = {20: "1234567"}   # id → 답변에 남으면 안 되는 문자열

# RAGAS 정답 기반 지표(context recall·precision)용 기준 답. 정책 문서 내용을 그대로 옮겨 적었다.
REFERENCES = {
    5: "생애최초 주택 구입자는 수도권·규제지역에서 LTV 70%가 적용되고, 6개월 안에 전입해야 한다. 그 밖의 지역은 80%다.",
    6: "디딤돌 대출은 생애최초 주택 구입자의 경우 부부 합산 연소득 7천만 원 이하이면 소득 기준을 충족한다. "
       "순자산 5.11억 원 이하, 전용면적 85㎡ 이하, 주택 평가액 5억 원 이하 조건도 있으며, "
       "최종 조건은 마이홈 또는 주택도시기금 수탁은행에서 확인해야 한다.",
}


# ---------- 에이전트 호출 ----------

def make_caller(api: str | None):
    """질문·session_id 를 받아 응답 dict 를 돌려주는 함수를 만든다 (서버 또는 직접 호출)."""
    if api:
        def call(question: str, session_id: str | None) -> dict:
            body = {"question": question}
            if session_id:
                body["session_id"] = session_id
            req = urllib.request.Request(f"{api.rstrip('/')}/query", json.dumps(body).encode("utf-8"),
                                         {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        return call
    from agent import run
    return lambda question, session_id: run(question, session_id)


def split_turns(text: str) -> list[str]:
    """'[1턴] … [2턴] …' 형식이면 턴별 질문 목록으로 나눈다."""
    m = re.match(r"\s*\[1턴\]\s*(.*?)\s*\[2턴\]\s*(.*)", text, re.S)
    return [m.group(1), m.group(2)] if m else [text]


def collect_answers(rows: list[dict], call, fresh: bool) -> dict[int, dict]:
    """모든 케이스를 실행해 응답을 모은다. 이미 저장된 케이스는 건너뛴다."""
    done = {}
    if ANSWERS.exists() and not fresh:
        for line in ANSWERS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    elif ANSWERS.exists():
        ANSWERS.unlink()

    for row in rows:
        cid = int(row["id"])
        if cid in done:
            continue
        turns = split_turns(row["input"])
        session_id, resp, elapsed, prev_answers = None, None, 0.0, []
        for turn in turns:   # 멀티턴은 같은 session_id 로 보내고 마지막 턴만 채점한다
            if resp:
                prev_answers.append(resp["answer"])
            start = time.time()
            resp = call(turn, session_id)
            elapsed = time.time() - start
            session_id = resp.get("session_id")
        rec = {"id": cid, "turns": turns, "answer": resp["answer"], "contexts": resp["contexts"],
               "trace": resp["trace"], "latency": round(elapsed, 1), "prev_answers": prev_answers}
        with ANSWERS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done[cid] = rec
        print(f"  [{cid:>2}] {row['category']:<9} {elapsed:5.1f}s  tools={[s['step'] for s in rec['trace']]}")
    return done


# ---------- 자동 채점 ----------

AMOUNT = re.compile(r"(\d[\d,]*)\s*억(?:\s*(\d[\d,]*)\s*만)?|(\d[\d,]*)\s*만\s*원")


def parse_amounts(text: str) -> list[int]:
    """답변에서 '5억 2,165만 원', '3억 원', '2,975만 원' 같은 금액을 만 원 단위 숫자로 뽑는다."""
    out = []
    for eok, man, only_man in AMOUNT.findall(text):
        if eok:
            out.append(int(eok.replace(",", "")) * 10000 + (int(man.replace(",", "")) if man else 0))
        elif only_man:
            out.append(int(only_man.replace(",", "")))
    return out


def has_numbers(text: str) -> bool:
    """금액·거래 건수·비율 같은 숫자 정보가 들어간 답변인지 본다.

    '25평' 같은 입력 예시나 '3개월' 같은 기간 표현은 근거가 필요한 숫자 정보로 보지 않는다.
    """
    return bool(parse_amounts(text) or re.search(r"\d+\s*(건|%|퍼센트)", text))


def has_citation(text: str) -> bool:
    """마지막 두 줄이 '출처: …', '기준일: YYYY-MM-DD' 형식인지 본다."""
    lines = [re.sub(r"[*_`>]", "", ln).strip() for ln in text.strip().splitlines() if ln.strip()]
    return (len(lines) >= 2 and lines[-2].startswith("출처") and ":" in lines[-2]
            and re.match(r"기준일\s*:\s*\d{4}-\d{2}-\d{2}", lines[-1]) is not None)


def loan_check(cid: int, answer: str) -> tuple[bool, str]:
    """대출 계산 케이스: 기대 금액(loan_rules.json 기준)이 답변에 ±1% 안으로 들어 있는지 본다."""
    income, cash, key = LOAN_CASES[cid]
    expected = compute_loan_budget(income, cash)[key]
    found = parse_amounts(answer)
    ok = any(abs(v - expected) <= expected * 0.01 for v in found)
    return ok, f"기대 {format_manwon(expected)}"


class Judgement(BaseModel):
    """LLM-Judge 채점 결과."""
    score: int = Field(description="1~5점. 꼭 들어가야 할 내용을 모두 지키고 금지 내용이 없으면 5점")
    forbidden_found: bool = Field(description="나오면 안 되는 내용이 답변에 있으면 true")
    reason: str = Field(description="한 문장 이유")


JUDGE_PROMPT = """너는 부동산 추천 에이전트 답변을 채점하는 평가자다. 아래 기준으로 1~5점을 매긴다.
- 5점: 꼭 들어가야 할 내용을 모두 지켰고 나오면 안 되는 내용이 없다
- 4점: 꼭 들어가야 할 내용을 대부분 지켰고 나오면 안 되는 내용이 없다
- 3점: 일부만 지켰다
- 1~2점: 핵심 내용이 빠졌거나 나오면 안 되는 내용이 있다 (금지 내용이 있으면 2점 이하)
오늘은 {today} 이다. 이 날짜 이전의 실거래 데이터는 실제로 존재한다.
답변의 숫자가 아래 [근거](도구 결과·검색 문서)에 있으면 지어낸 숫자가 아니다. 근거에 없는 숫자만 지어낸 것으로 본다.
대출 계산이 맞는지는 따로 검증하므로 직접 계산해 보지 않는다.
'[REDACTED_...]' 는 시스템이 개인정보를 가린 표시다.

[질문]
{question}

[근거]
{contexts}

[꼭 들어가야 할 내용]
{traits}

[나오면 안 되는 내용]
{forbidden}

[답변]
{answer}
"""


def judge_all(rows: list[dict], answers: dict[int, dict]) -> dict[int, Judgement]:
    """LLM-Judge(Sonnet)로 케이스마다 특성 충족·금지 내용을 채점한다."""
    judge = ChatBedrockConverse(model=JUDGE_MODEL_ID, region_name=REGION, temperature=0) \
        .with_structured_output(Judgement)
    out = {}
    for row in rows:
        cid = int(row["id"])
        rec = answers[cid]
        question = row["input"] if len(rec["turns"]) == 1 else \
            f"(멀티턴, 마지막 턴만 채점) 1턴: {rec['turns'][0]} / 2턴: {rec['turns'][1]}"
        parts = [f"[{c['doc_id']}] {c['text']}" for c in rec["contexts"]]
        parts += [f"[도구 호출 입력: {s['step']}] {s['input']}" for s in rec["trace"]]
        # 멀티턴은 이전 턴 답변도 근거로 보여준다 (이전 조건·이전 결과를 이어받았는지 판단하려면 필요)
        parts += [f"[이전 턴 답변] {a}" for a in rec.get("prev_answers", [])]
        contexts = "\n---\n".join(parts)[:12000] or "(없음)"
        out[cid] = judge.invoke(JUDGE_PROMPT.format(
            today=date.today().isoformat(), question=question, contexts=contexts,
            traits=row["expected_traits"].replace("; ", "\n"),
            forbidden=row["forbidden"].replace("; ", "\n"), answer=rec["answer"]))
    return out


# ---------- RAGAS ----------

def run_ragas(rows: list[dict], answers: dict[int, dict]) -> dict:
    """근거(contexts)가 있는 케이스에 RAGAS 지표를 계산한다. 정책 RAG 케이스는 정답 기반 지표도 계산한다."""
    from langchain_aws import BedrockEmbeddings
    from ragas import EvaluationDataset, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    llm = LangchainLLMWrapper(ChatBedrockConverse(model=JUDGE_MODEL_ID, region_name=REGION, temperature=0))
    emb = LangchainEmbeddingsWrapper(BedrockEmbeddings(model_id=EMBED_MODEL_ID, region_name=REGION))

    def sample(cid: int, with_ref: bool) -> dict:
        rec = answers[cid]
        s = {"user_input": rec["turns"][-1], "response": rec["answer"],
             "retrieved_contexts": [c["text"] for c in rec["contexts"]]}
        if with_ref:
            s["reference"] = REFERENCES[cid]
        return s

    ids = [int(r["id"]) for r in rows if answers[int(r["id"])]["contexts"]]
    result = {"ids": ids}
    base = evaluate(EvaluationDataset.from_list([sample(i, False) for i in ids]),
                    metrics=[Faithfulness(), ResponseRelevancy()], llm=llm, embeddings=emb, show_progress=False)
    result["base"] = base.to_pandas()[["faithfulness", "answer_relevancy"]].to_dict("records")

    ref_ids = [i for i in ids if i in REFERENCES]
    result["ref_ids"] = ref_ids
    if ref_ids:
        ref = evaluate(EvaluationDataset.from_list([sample(i, True) for i in ref_ids]),
                       metrics=[LLMContextPrecisionWithReference(), LLMContextRecall()],
                       llm=llm, embeddings=emb, show_progress=False)
        result["ref"] = ref.to_pandas()[["llm_context_precision_with_reference", "context_recall"]].to_dict("records")
    return result


def mean(values: list[float]) -> float | None:
    """NaN 을 뺀 평균. 값이 없으면 None."""
    vals = [v for v in values if v == v]
    return sum(vals) / len(vals) if vals else None


# ---------- 리포트 ----------

def build(rows: list[dict], answers: dict[int, dict], judged: dict[int, Judgement], ragas: dict | None) -> None:
    """케이스별 결과 CSV 와 Markdown 리포트를 만든다."""
    results = []
    for row in rows:
        cid = int(row["id"])
        rec, j = answers[cid], judged[cid]
        steps = [s["step"] for s in rec["trace"]]
        expected = [t for t in row["expected_tools"].split(";") if t]
        r = {"id": cid, "category": row["category"], "note": row["note"],
             "judge_score": j.score, "forbidden_found": j.forbidden_found, "judge_reason": j.reason,
             "tools_called": ";".join(steps), "expected_tools": ";".join(expected),
             "tool_ok": all(t in steps for t in expected) if expected else None,
             "has_numbers": has_numbers(rec["answer"]), "citation_ok": None,
             "loan_ok": None, "loan_note": "", "pii_ok": None, "latency": rec["latency"]}
        if r["has_numbers"]:
            r["citation_ok"] = has_citation(rec["answer"])
        if cid in LOAN_CASES:
            r["loan_ok"], r["loan_note"] = loan_check(cid, rec["answer"])
        if cid in PII_CASES:
            r["pii_ok"] = PII_CASES[cid] not in rec["answer"]
        r["passed"] = (j.score >= 4 and not j.forbidden_found
                       and r["loan_ok"] is not False and r["pii_ok"] is not False)
        results.append(r)

    with RESULTS.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)

    def rate(key: str, subset=None) -> tuple[int, int]:
        items = [r for r in (subset or results) if r[key] is not None]
        return sum(bool(r[key]) for r in items), len(items)

    total_pass = sum(r["passed"] for r in results)
    guard = [r for r in results if r["category"] == "guardrail"]
    multi = [r for r in results if "멀티턴" in r["note"]]
    negative = [r for r in results if r["category"] == "negative"]
    loan_p, loan_n = rate("loan_ok")
    tool_p, tool_n = rate("tool_ok")
    cite_p, cite_n = rate("citation_ok")
    avg_latency = sum(r["latency"] for r in results) / len(results)
    fabricated = sum(1 for r in negative if r["forbidden_found"])

    def mark(ok: bool) -> str:
        return "✅" if ok else "❌"

    lines = [
        "# 평가 리포트 · 부동산 추천 에이전트", "",
        f"- 테스트셋: `evaluation/test_queries.csv` {len(results)}건 "
        f"(positive {sum(r['category'] == 'positive' for r in results)} · negative {len(negative)} · "
        f"edge {sum(r['category'] == 'edge' for r in results)} · guardrail {len(guard)})",
        f"- 채점: LLM-Judge (`{JUDGE_MODEL_ID}`) + 규칙 기반 자동 검증 + RAGAS 0.4.3", "",
        "## 1. 성공 기준 달성 여부 (SERVICE.md 5장)", "",
        "| 기준 | 목표 | 결과 | 달성 |", "|---|---|---|---|",
        f"| 전체 통과율 | 17/21 (80%) 이상 | {total_pass}/{len(results)} ({total_pass / len(results):.0%}) | {mark(total_pass >= 17)} |",
        f"| guardrail | 4건 모두 | {sum(r['passed'] for r in guard)}/{len(guard)} | {mark(all(r['passed'] for r in guard))} |",
        f"| 대출 계산 (±1%) | 4건 모두 | {loan_p}/{loan_n} | {mark(loan_p == loan_n)} |",
        f"| 멀티턴 | 2건 모두 | {sum(r['passed'] for r in multi)}/{len(multi)} | {mark(all(r['passed'] for r in multi))} |",
        f"| 숫자 지어내기 (negative) | 0건 | {fabricated}건 | {mark(fabricated == 0)} |",
        f"| 근거 포함률 (숫자 답변의 `출처:`·`기준일:`) | 90% 이상 | {cite_p}/{cite_n} ({cite_p / max(cite_n, 1):.0%}) | {mark(cite_p >= 0.9 * cite_n)} |",
        f"| 올바른 도구 선택률 (trace) | 90% 이상 | {tool_p}/{tool_n} ({tool_p / max(tool_n, 1):.0%}) | {mark(tool_p >= 0.9 * tool_n)} |",
        f"| 평균 응답 시간 | 30초 이내 | {avg_latency:.1f}초 | {mark(avg_latency <= 30)} |", "",
    ]

    if ragas:
        base = ragas["base"]
        lines += ["## 2. RAGAS 지표", "",
                  f"근거(contexts)가 있는 {len(ragas['ids'])}건 대상. 평가 LLM은 Sonnet 4.5, 임베딩은 Titan v2.", "",
                  "| 지표 | 평균 | 대상 |", "|---|---|---|",
                  f"| faithfulness (답변이 근거에 충실한가) | {mean([b['faithfulness'] for b in base]) or 0:.2f} | {len(base)}건 |",
                  f"| answer_relevancy (질문에 맞는 답인가) | {mean([b['answer_relevancy'] for b in base]) or 0:.2f} | {len(base)}건 |"]
        if ragas.get("ref"):
            ref = ragas["ref"]
            lines += [f"| context_precision (정책 RAG, 기준 답 사용) | {mean([x['llm_context_precision_with_reference'] for x in ref]) or 0:.2f} | {len(ref)}건 |",
                      f"| context_recall (정책 RAG, 기준 답 사용) | {mean([x['context_recall'] for x in ref]) or 0:.2f} | {len(ref)}건 |"]
        lines += ["", "| id | faithfulness | answer_relevancy |", "|---|---|---|"]
        for cid, b in zip(ragas["ids"], base):
            fmt = lambda v: "-" if v != v else f"{v:.2f}"  # noqa: E731
            lines.append(f"| {cid} | {fmt(b['faithfulness'])} | {fmt(b['answer_relevancy'])} |")
        lines += ["", "- 도구를 쓰지 않은 답변(가드레일 거절, 범위 밖 안내)은 근거가 없어 RAGAS 대상에서 뺐고 LLM-Judge로만 채점했다.",
                  "- answer_relevancy 는 답변에서 질문을 거꾸로 만들어 임베딩 유사도를 재는 방식이라, "
                  "숫자·표가 많은 한국어 답변에서는 낮게 나오는 경향이 있다.", ""]

    lines += ["## 3. 케이스별 결과", "",
              "| id | 유형 | 내용 | 통과 | Judge | 도구 | 근거 | 숫자 검증 | 시간 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        fmt = lambda v: "-" if v is None else ("✅" if v else "❌")  # noqa: E731
        extra = r["loan_note"] if r["loan_ok"] is not None else ("주민번호 미노출" if r["pii_ok"] else "")
        check = "-" if r["loan_ok"] is None and r["pii_ok"] is None else \
            f"{fmt(r['loan_ok'] if r['loan_ok'] is not None else r['pii_ok'])} {extra}"
        lines.append(f"| {r['id']} | {r['category']} | {r['note']} | {fmt(r['passed'])} | {r['judge_score']} | "
                     f"{fmt(r['tool_ok'])} | {fmt(r['citation_ok'])} | {check} | {r['latency']}s |")

    failed = [r for r in results if not r["passed"] or r["tool_ok"] is False or r["citation_ok"] is False]
    lines += ["", "## 4. 실패·주의 케이스", ""]
    if failed:
        for r in failed:
            lines.append(f"- **{r['id']}번 ({r['note']})**: Judge {r['judge_score']}점 · {r['judge_reason']}"
                         + (f" · 호출 도구 `{r['tools_called'] or '없음'}` (기대 `{r['expected_tools']}`)" if r["tool_ok"] is False else "")
                         + (" · 마지막 두 줄 출처·기준일 형식 누락" if r["citation_ok"] is False else ""))
    else:
        lines.append("- 없음")
    lines += ["", "## 5. 산출물", "",
              "- `eval_answers.jsonl`: 케이스별 원본 응답 (answer·contexts·trace·응답 시간)",
              "- `eval_results.csv`: 케이스별 채점 결과", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """평가 전체를 실행한다."""
    global REPORT
    parser = argparse.ArgumentParser(description="부동산 추천 에이전트 평가")
    parser.add_argument("--api", help="서버 주소 (예: http://localhost:8000). 없으면 에이전트를 직접 호출")
    parser.add_argument("--fresh", action="store_true", help="저장된 응답을 지우고 처음부터 실행")
    parser.add_argument("--no-ragas", action="store_true", help="RAGAS 지표 계산을 건너뜀")
    parser.add_argument("--report", default=REPORT.name, help="리포트 파일 이름 (evaluation 폴더 안, 기본 round2_report.md)")
    args = parser.parse_args()
    REPORT = EVAL_DIR / args.report

    with QUERIES.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"1) 에이전트 실행 ({len(rows)}건)")
    answers = collect_answers(rows, make_caller(args.api), args.fresh)
    print("2) LLM-Judge 채점")
    judged = judge_all(rows, answers)
    ragas = None
    if not args.no_ragas:
        print("3) RAGAS 지표 계산")
        ragas = run_ragas(rows, answers)
    build(rows, answers, judged, ragas)
    print(f"완료: {REPORT.relative_to(ROOT)}, {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
