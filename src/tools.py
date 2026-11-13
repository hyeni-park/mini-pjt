"""부동산 추천 에이전트 도메인 도구 모음.

지역 조회, 실거래 조회·요약, 대출 계산, 정책 문서 검색, 계산 조건 저장·불러오기·삭제 도구를 정의한다.
숫자 계산은 모두 이 파일 안에서 하고, 대출 기준 숫자는 data/loan_rules.json 에서만 읽는다.
"""
import json
import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
# LangSmith 트레이스에는 실행 흐름만 남기고 입력·출력 내용(연소득·현금 등)은 보내지 않는다 (SERVICE.md 개인정보 정책)
# 내용을 봐야 할 때만 .env 에 false 로 적어 끈다
os.environ.setdefault("LANGSMITH_HIDE_INPUTS", "true")
os.environ.setdefault("LANGSMITH_HIDE_OUTPUTS", "true")

DATA = ROOT / "data"
RULES = json.loads((DATA / "loan_rules.json").read_text(encoding="utf-8"))
REGIONS = json.loads((DATA / "regions.json").read_text(encoding="utf-8"))

TRADE_URL = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
TRADE_SOURCE = "국토교통부 아파트 매매 실거래가 공개 API"
LOAN_SOURCE = "data/loan_rules.json (금융위원회 발표 등 공개 자료 기반 가정값)"
PYEONG_TO_M2 = 3.3058
EXCLUSIVE_RATIO = 0.75   # 공급면적 대비 전용면적 비율 (가정)
AREA_TOLERANCE = 5.0     # 전용면적 허용 오차 (㎡)
TOP_N = 5


def _today() -> date:
    """조회 기준일(오늘 날짜)을 돌려준다."""
    return date.today()


def _recent_months(n: int = 3) -> list[str]:
    """이번 달을 제외한 직전 n개월을 YYYYMM 문자열 목록(과거 → 최근)으로 돌려준다."""
    y, m = _today().year, _today().month
    months = []
    for _ in range(n):
        m -= 1
        if m == 0:
            y, m = y - 1, 12
        months.append(f"{y}{m:02d}")
    return list(reversed(months))


def _find_region(name: str) -> str | None:
    """지역명이나 별칭으로 regions.json 의 정식 지역명을 찾는다. 없으면 None."""
    key = name.replace(" ", "")
    for full, info in REGIONS["regions"].items():
        names = [full] + info["aliases"]
        if any(key == n.replace(" ", "") for n in names):
            return full
    return None


def _pyeong_to_area(pyeong: float) -> float:
    """공급면적 기준 평형을 전용면적(㎡)으로 환산한다."""
    return round(pyeong * PYEONG_TO_M2 * EXCLUSIVE_RATIO, 1)


@lru_cache(maxsize=256)
def _fetch_trades(code: str, ym: str) -> tuple[int, tuple]:
    """국토부 실거래가 API에서 한 지역·한 달의 거래를 모두 가져온다 (numOfRows=1000).

    반환값은 (전체 건수, 거래 튜플) 이다. 같은 요청은 메모리에 캐시한다.
    API 가 오류 코드를 돌려주면 예외를 낸다 (예외는 캐시되지 않아 다음 요청에서 다시 조회한다).
    """
    query = urllib.parse.urlencode({
        "serviceKey": os.environ["DATA_GO_KR_API_KEY"],
        "LAWD_CD": code, "DEAL_YMD": ym, "numOfRows": 1000, "pageNo": 1,
    })
    with urllib.request.urlopen(f"{TRADE_URL}?{query}", timeout=30) as resp:
        root = ET.fromstring(resp.read())
    result_code = root.findtext("header/resultCode")
    if result_code not in ("000", "00"):
        reason = root.findtext("header/resultMsg") or root.findtext(".//returnAuthMsg") or "알 수 없는 응답"
        raise RuntimeError(f"실거래가 API 오류 ({result_code}): {reason}")
    items = []
    for it in root.findall("body/items/item"):
        items.append((
            (it.findtext("aptNm") or "").strip(),
            (it.findtext("umdNm") or "").strip(),
            int((it.findtext("dealAmount") or "0").replace(",", "")),
            float(it.findtext("excluUseAr") or 0),
            bool((it.findtext("cdealType") or "").strip()),
        ))
    return int(root.findtext("body/totalCount") or 0), tuple(items)


def _try_fetch(code: str, ym: str) -> tuple[int, tuple] | None:
    """실거래를 조회하되, 네트워크·API 오류면 None 을 돌려준다 (도구가 '조회 실패'로 답하게 한다)."""
    try:
        return _fetch_trades(code, ym)
    except Exception:
        return None


def _annuity_pv(payment: float, rate: float, months: int) -> float:
    """월 상환액·연이율·개월 수로 원리금균등 대출 원금을 역산한다."""
    r = rate / 12
    return payment * (1 - (1 + r) ** -months) / r


def _annuity_payment(principal: float, rate: float, months: int) -> float:
    """원리금균등 상환 월 납입액을 계산한다."""
    r = rate / 12
    return principal * r / (1 - (1 + r) ** -months)


def format_manwon(manwon: float) -> str:
    """만 원 단위 금액을 '5억 2,165만 원' 같은 한국어 표기로 바꾼다."""
    v = round(manwon)
    eok, man = divmod(v, 10000)
    if eok and man:
        return f"{eok}억 {man:,}만 원"
    if eok:
        return f"{eok}억 원"
    return f"{man:,}만 원"


def compute_loan_budget(annual_income_manwon: float, cash_manwon: float) -> dict:
    """연소득·보유 현금(만 원)으로 대출 한도와 매수 가능 금액을 계산한다 (평가 스크립트에서도 사용)."""
    ltv = RULES["ltv"]["rate"]
    dsr = RULES["dsr"]["limit"]
    cap = RULES["max_loan_amount"]["amount_manwon"]
    rate = RULES["loan"]["rate"]
    stress = RULES["stress_rate"]["add_rate"]
    months = RULES["loan"]["term_years"] * 12

    dsr_loan = _annuity_pv(annual_income_manwon * dsr / 12, rate + stress, months)
    candidates = {
        "LTV": cash_manwon / (1 - ltv),
        "DSR": cash_manwon + dsr_loan,
        "총액 한도": cash_manwon + cap,
    }
    binding = min(candidates, key=candidates.get)
    max_price = candidates[binding]
    loan = max_price - cash_manwon
    return {
        "매수_가능_금액_만원": round(max_price),
        "대출_한도_만원": round(loan),
        "월_상환액_만원": round(_annuity_payment(loan, rate, months)),
        "한도를_정한_규제": binding,
        "규제별_매수_가능_금액_만원": {k: round(v) for k, v in candidates.items()},
    }


@tool
def region_code_lookup(region_name: str) -> str:
    """지역명(예: 송파구, 수지, 용인 수지구)을 법정동 시군구 코드(5자리)로 바꾼다.

    지원 지역이 아니면 지원하지 않는다고 알려준다.
    """
    full = _find_region(region_name)
    if not full:
        return json.dumps({"결과": "지원하지 않는 지역", "지원_지역": list(REGIONS["regions"])}, ensure_ascii=False)
    return json.dumps({"지역": full, "시군구_코드": REGIONS["regions"][full]["code"]}, ensure_ascii=False)


@tool
def get_candidate_regions(workplace: str) -> str:
    """직장 위치(잠실·판교·분당)를 받아 통근 가능한 후보 지역 목록을 돌려준다.

    지원하지 않는 직장 위치면 지원 거점 목록을 알려준다.
    """
    key = workplace.replace(" ", "").replace("역", "")
    for name, regions in REGIONS["workplaces"].items():
        if name in key:
            return json.dumps({"직장_위치": name, "후보_지역": regions,
                               "참고": "대중교통 약 1시간 이내 통근을 가정한 정적 목록"}, ensure_ascii=False)
    return json.dumps({"결과": "지원하지 않는 직장 위치", "지원_거점": list(REGIONS["workplaces"])}, ensure_ascii=False)


@tool
def apt_trades(region_name: str, year_month: str) -> str:
    """특정 지역·연월(YYYYMM)의 아파트 매매 실거래를 단순 조회한다.

    거래 건수(해제 거래 포함·제외)와 가격 범위만 돌려준다. 거래가 없으면 '조회 결과 없음'.
    """
    full = _find_region(region_name)
    if not full:
        return json.dumps({"결과": "지원하지 않는 지역"}, ensure_ascii=False)
    ym = year_month.replace("-", "").replace(".", "")[:6]
    fetched = _try_fetch(REGIONS["regions"][full]["code"], ym)
    if fetched is None:
        return json.dumps({"지역": full, "연월": ym, "결과": "조회 실패",
                           "안내": "실거래가 API 응답 오류. 잠시 후 다시 시도"}, ensure_ascii=False)
    total, items = fetched
    valid = [p for _, _, p, _, c in items if not c]
    if total == 0 or not valid:
        return json.dumps({"지역": full, "연월": ym, "결과": "조회 결과 없음",
                           "출처": TRADE_SOURCE, "기준일": _today().isoformat()}, ensure_ascii=False)
    return json.dumps({
        "지역": full, "연월": ym,
        "전체_거래_건수": total,
        "해제_제외_거래_건수": len(valid),
        "평균가_만원": round(sum(valid) / len(valid)),
        "최저가_만원": min(valid), "최고가_만원": max(valid),
        "출처": TRADE_SOURCE, "기준일": _today().isoformat(),
    }, ensure_ascii=False)


@tool
def summarize_trades(regions: list[str], pyeong: float, max_price_manwon: float | None = None) -> str:
    """후보 지역들의 직전 3개월 실거래를 직접 조회해 요약한다.

    - regions: 지역명 목록 (예: ["용인 수지구", "용인 기흥구"])
    - pyeong: 희망 평형 (공급면적 기준, 예: 25). 전용면적으로 환산해 ±5㎡ 거래만 쓴다.
    - max_price_manwon: 매수 가능 금액(만 원). 주면 평균가가 이 금액 이하인 단지를
      거래 건수 순으로 최대 5개 돌려주고, 없으면 지역별 시세 요약만 돌려준다.
    해제 거래는 제외한다. 거래 원본은 돌려주지 않는다.
    """
    months = _recent_months()
    target = _pyeong_to_area(pyeong)
    region_summary, groups, unknown, failed = {}, defaultdict(list), [], []
    for name in regions:
        full = _find_region(name)
        if not full:
            unknown.append(name)
            continue
        fetched = [_try_fetch(REGIONS["regions"][full]["code"], ym) for ym in months]
        if any(f is None for f in fetched):   # 한 달이라도 실패하면 그 지역은 요약하지 않는다
            failed.append(full)
            region_summary[full] = {"결과": "조회 실패"}
            continue
        prices = []
        for _, items in fetched:
            for apt, dong, price, area, cancelled in items:
                if cancelled or abs(area - target) > AREA_TOLERANCE:
                    continue
                prices.append(price)
                groups[(full, dong, apt)].append(price)
        region_summary[full] = ({"거래_건수": len(prices), "평균가_만원": round(sum(prices) / len(prices)),
                                 "최저가_만원": min(prices), "최고가_만원": max(prices)}
                                if prices else {"거래_건수": 0, "결과": "조회 결과 없음"})

    result = {
        "조회_기간": f"{months[0][:4]}-{months[0][4:]} ~ {months[-1][:4]}-{months[-1][4:]}",
        "평형_조건": f"{pyeong}평 → 전용 {target}㎡ ±{AREA_TOLERANCE:g}㎡",
        "지역별_요약": region_summary,
    }
    if unknown:
        result["지원하지_않는_지역"] = unknown
    if failed:
        result["조회_실패_지역"] = failed
        result["안내"] = "실거래가 API 응답 오류로 일부 지역을 조회하지 못함. 잠시 후 다시 시도"
    if max_price_manwon is not None:
        rows = [{"지역": k[0], "동": k[1], "단지": k[2], "거래_건수": len(v),
                 "평균가_만원": round(sum(v) / len(v)), "최저가_만원": min(v), "최고가_만원": max(v)}
                for k, v in groups.items() if sum(v) / len(v) <= max_price_manwon]
        rows.sort(key=lambda r: (-r["거래_건수"], r["평균가_만원"]))
        result["매수_가능_금액_만원"] = round(max_price_manwon)
        result["예산_이하_단지_수"] = len(rows)
        result["추천_단지"] = rows[:TOP_N] if rows else ("조회 실패" if failed else "조회 결과 없음")
    result["출처"] = TRADE_SOURCE
    result["기준일"] = _today().isoformat()
    return json.dumps(result, ensure_ascii=False)


@tool
def calc_loan_budget(annual_income_manwon: float, cash_manwon: float) -> str:
    """연소득(세전, 만 원, 부부는 합산)과 보유 현금(만 원)으로 대출 한도·월 상환액·매수 가능 금액을 계산한다.

    생애최초 주택 구입자 기준으로 계산한다.
    LTV·DSR(스트레스 금리 반영)·총액 한도 중 가장 작은 값이 한도가 된다. 기존 부채·부대비용은 반영하지 않는다.
    예: 연봉 7천 → 7000, 현금 2억 → 20000
    """
    out = compute_loan_budget(annual_income_manwon, cash_manwon)
    out["표기"] = {
        "매수_가능_금액": format_manwon(out["매수_가능_금액_만원"]),
        "대출_한도": format_manwon(out["대출_한도_만원"]),
        "월_상환액": format_manwon(out["월_상환액_만원"]),
    }
    out["입력"] = {"연소득_만원": annual_income_manwon, "보유_현금_만원": cash_manwon}
    out["계산_기준"] = {
        "LTV": RULES["ltv"]["rate"], "DSR": RULES["dsr"]["limit"],
        "스트레스_가산금리": RULES["stress_rate"]["add_rate"],
        "총액_한도_만원": RULES["max_loan_amount"]["amount_manwon"],
        "금리": RULES["loan"]["rate"], "만기_년": RULES["loan"]["term_years"], "상환": "원리금균등",
    }
    out["계산_가정"] = (f"{RULES['meta']['scope']} 기준 (LTV {RULES['ltv']['rate']:.0%}). "
                      "생애최초 구입자가 아니면 LTV가 낮아 매수 가능 금액이 줄어들 수 있음")
    out["면책"] = RULES["meta"]["disclaimer"]
    out["출처"] = LOAN_SOURCE
    out["기준일"] = RULES["meta"]["basis_date"]
    return json.dumps(out, ensure_ascii=False)


@tool
def search_policy(query: str) -> str:
    """정책 대출(디딤돌·보금자리론)과 대출 규제(LTV·DSR·총액 한도·스트레스 DSR) 문서를 검색한다.

    관련 문서 조각과 출처(doc_id)를 돌려준다.
    """
    from retriever import search
    return json.dumps(search(query), ensure_ascii=False)


MEMORY_NAMESPACE = "users"
MEMORY_KEY = "conditions"


@tool
def save_conditions(annual_income_manwon: int, cash_manwon: int, workplace: str, pyeong: float,
                    runtime: ToolRuntime) -> str:
    """사용자가 저장을 요청했을 때 계산 조건(연소득·보유 현금·직장 위치·희망 평형)을 장기 메모리에 저장한다.

    실행 전에 사용자 승인(HITL)을 받는다. 대화 내용이나 주민등록번호 등은 저장하지 않는다.
    """
    user_id = runtime.context.user_id if runtime.context else None
    if not user_id or runtime.store is None:
        return json.dumps({"결과": "저장 불가", "이유": "사용자 식별 정보(user_id)가 없습니다"}, ensure_ascii=False)
    value = {"annual_income_manwon": annual_income_manwon, "cash_manwon": cash_manwon,
             "workplace": workplace, "pyeong": pyeong, "saved_at": _today().isoformat()}
    runtime.store.put((MEMORY_NAMESPACE, user_id), MEMORY_KEY, value)
    return json.dumps({"결과": "저장 완료", "저장한_조건": value}, ensure_ascii=False)


@tool
def load_conditions(runtime: ToolRuntime) -> str:
    """사용자가 '저번 조건', '저장한 조건'을 요청했을 때 장기 메모리에서 저장된 계산 조건을 불러온다."""
    user_id = runtime.context.user_id if runtime.context else None
    if not user_id or runtime.store is None:
        return json.dumps({"결과": "불러오기 불가", "이유": "사용자 식별 정보(user_id)가 없습니다"}, ensure_ascii=False)
    item = runtime.store.get((MEMORY_NAMESPACE, user_id), MEMORY_KEY)
    if item is None:
        return json.dumps({"결과": "저장된 조건 없음"}, ensure_ascii=False)
    return json.dumps({"결과": "불러오기 완료", "저장된_조건": item.value}, ensure_ascii=False)


@tool
def delete_conditions(runtime: ToolRuntime) -> str:
    """사용자가 저장한 조건을 지워 달라고 요청했을 때 장기 메모리에서 삭제한다. 실행 전에 사용자 승인(HITL)을 받는다."""
    user_id = runtime.context.user_id if runtime.context else None
    if not user_id or runtime.store is None:
        return json.dumps({"결과": "삭제 불가", "이유": "사용자 식별 정보(user_id)가 없습니다"}, ensure_ascii=False)
    if runtime.store.get((MEMORY_NAMESPACE, user_id), MEMORY_KEY) is None:
        return json.dumps({"결과": "저장된 조건 없음"}, ensure_ascii=False)
    runtime.store.delete((MEMORY_NAMESPACE, user_id), MEMORY_KEY)
    return json.dumps({"결과": "삭제 완료"}, ensure_ascii=False)


MEMORY_TOOLS = {"save_conditions", "load_conditions", "delete_conditions"}

TOOLS = [region_code_lookup, get_candidate_regions, apt_trades, summarize_trades, calc_loan_budget, search_policy,
         save_conditions, load_conditions, delete_conditions]
