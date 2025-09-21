# rank_by_trends.py
import argparse, time, random, math, sys, csv, os
from typing import List, Tuple
import pandas as pd
from pytrends.request import TrendReq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

"""
정확한 유명도 비교 절차
- Google Trends는 한 번에 최대 5개까지만 상대 비교(0~100 정규화) 제공.
- 배치마다 공통 '앵커' 용어를 포함(예: "weather")하여 각 배치를 앵커 대비 비율로 변환.
- 모든 배치를 앵커 스케일로 환산하면 전원 공통 스케일에서 평균값 비교가 가능.
주의:
- 너무 강력한 앵커(검색량이 매우 큼)나 너무 약한 앵커(0이 자주 나옴)는 피하세요.
- 필요시 anchors를 여러 개 준비해 라운드로빈으로 배치에 넣을 수 있음.
"""

def read_names(csv_path: str) -> List[str]:
    df = pd.read_csv(csv_path)
    # 첫 번째 컬럼을 이름으로 사용 (예: 'actor')
    name_col = df.columns[0]
    names = (
        df[name_col]
        .astype(str)
        .str.strip()
        .tolist()
    )
    # 헤더 행 제거(파일 첫 줄이 'actor' 같은 헤더인 경우)
    if len(names) > 0 and names[0].lower() == name_col.lower():
        names = names[1:]
    # 빈 문자열 제거, 중복 제거(원 순서 유지)
    seen = set()
    uniq = []
    for n in names:
        if n and n not in seen:
            seen.add(n)
            uniq.append(n)
    return uniq

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30),
       retry=retry_if_exception_type(Exception))
def fetch_batch(pytrends: TrendReq, terms: List[str], timeframe: str = "today 5-y", geo: str = "") -> pd.DataFrame:
    pytrends.build_payload(terms, timeframe=timeframe, geo=geo, gprop="")
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        raise RuntimeError("Empty Trends response")
    # 마지막 'isPartial' 컬럼 제거
    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])
    return df

def calc_anchor_norm_means(df: pd.DataFrame, anchor: str) -> Tuple[pd.Series, float]:
    # 각 용어의 5년 평균
    means = df.mean(axis=0)
    anchor_mean = float(means.get(anchor, float("nan")))
    if anchor_mean == 0 or math.isnan(anchor_mean):
        # 앵커가 0 또는 NaN이면 배치를 버리고 다시 시도하는 것이 안전
        raise RuntimeError("Anchor has zero/NaN mean; choose a different anchor or retry")
    # 앵커 대비 비율(배치 내 정규화)
    norm = means / anchor_mean
    return norm, anchor_mean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="/home/ldlqudgus756/EraseAtScale/EAS/EAS_sd1.4/configs/train_celeb/prompt_target.csv", \
                     help="CSV with one column of names (e.g., 'actor')")
    ap.add_argument("--out", default="ranked_celebrities.csv")
    ap.add_argument("--timeframe", default="today 5-y", help="e.g., 'today 5-y'")
    ap.add_argument("--geo", default="", help="country code like 'US' or '' for worldwide")
    ap.add_argument("--sleep", type=float, default=2.5, help="seconds between requests to avoid 429")
    ap.add_argument("--anchor", default="weather", help="anchor term included in every batch")
    ap.add_argument("--rotate_anchors", action="store_true", help="use a small bank of anchors")
    args = ap.parse_args()

    names = read_names(args.targets)
    if not names:
        print("No names found.", file=sys.stderr)
        sys.exit(1)

    # 앵커 뱅크(옵션). 너무 강/약하지 않은 일반 단어들 제안.
    anchor_bank = ["weather", "news", "music", "movie", "internet"]
    anchor_iter = 0

    pytrends = TrendReq(hl="en-US", tz=0)  # 전 세계 기준

    records = []
    # Google Trends: 최대 5개 비교이므로 (앵커 1 + 셀럽 4)로 배치
    BATCH_N = 4

    global_anchor_means = []  # 배치별 앵커 평균(상대 크기 확인용)
    for idx, batch in enumerate(chunked(names, BATCH_N)):
        if idx%10 == 0:
            print(f"{idx}/{len(names)/BATCH_N}")

        anchor = args.anchor
        if args.rotate_anchors:
            anchor = anchor_bank[anchor_iter % len(anchor_bank)]
            anchor_iter += 1

        terms = [anchor] + batch
        # 중복 방지
        terms = list(dict.fromkeys(terms))
        try:
            df = fetch_batch(pytrends, terms, timeframe=args.timeframe, geo=args.geo)
            norm_means, anchor_mean = calc_anchor_norm_means(df, anchor)
            global_anchor_means.append(anchor_mean)
            for t in terms:
                if t == anchor:
                    continue
                # 배치 내 앵커 정규화된 평균값
                records.append({"concept": t, "score_mean_5y_norm": float(norm_means.get(t, 0.0))})
        except Exception as e:
            # 실패 시 해당 배치를 스킵하거나 재시도
            print(f"[WARN] Batch failed ({batch}): {e}", file=sys.stderr)
        time.sleep(args.sleep + random.random())

    out_df = pd.DataFrame(records)
    # 일부 배치 실패로 중복/결측이 있을 수 있음 -> 그룹 평균/최댓값으로 집계
    if out_df.empty:
        print("No data aggregated. Try different anchors or increase sleep.", file=sys.stderr)
        sys.exit(2)

    out_df = out_df.groupby("concept", as_index=False)["score_mean_5y_norm"].mean()

    # 순위 계산 (내림차순)
    out_df = out_df.sort_values("score_mean_5y_norm", ascending=False).reset_index(drop=True)
    out_df["rank"] = out_df.index + 1

    # 저장
    out_cols = ["rank", "concept", "score_mean_5y_norm"]
    out_df[out_cols].to_csv(args.out, index=False)
    print(f"Saved: {args.out} (rows={len(out_df)})")

if __name__ == "__main__":
    main()