import numpy as np
import pandas as pd


def classify_stop_types(
    stop_df: pd.DataFrame,
    morning_threshold: float = 25.0,
    evening_threshold: float = 25.0,
    low_usage_quantile: float = 0.25,
) -> pd.DataFrame:
    """출근형, 퇴근형, 출퇴근형, 생활형, 저이용형으로 정류소를 분류합니다."""
    if stop_df.empty:
        return stop_df
    df = stop_df.copy()
    if "boardings" not in df.columns:
        df["stop_type"] = "분류 불가"
        return df

    low_cutoff = df["boardings"].quantile(low_usage_quantile)

    def classify(row) -> str:
        """정류소 한 행의 집중도와 이용량을 기준으로 유형을 정합니다."""
        boardings = row.get("boardings", np.nan)
        morning = row.get("morning_concentration", np.nan)
        evening = row.get("evening_concentration", np.nan)
        if pd.notna(boardings) and boardings <= low_cutoff:
            return "저이용형"
        if pd.isna(morning) or pd.isna(evening):
            return "분류 불가"
        morning_high = morning >= morning_threshold
        evening_high = evening >= evening_threshold
        if morning_high and evening_high:
            return "출퇴근형"
        if morning_high and morning > evening:
            return "출근형"
        if evening_high and evening > morning:
            return "퇴근형"
        return "생활형"

    df["stop_type"] = df.apply(classify, axis=1)
    return df


def top_stops(stop_df: pd.DataFrame, metric: str = "boardings", n: int = 10) -> pd.DataFrame:
    """특정 지표 기준 상위 정류소를 반환합니다."""
    if stop_df.empty or metric not in stop_df.columns:
        return pd.DataFrame()
    return stop_df.sort_values(metric, ascending=False).head(n).copy()


def district_totals(stop_df: pd.DataFrame, metric: str = "boardings") -> pd.DataFrame:
    """구·군별 지표 합계를 계산합니다."""
    if stop_df.empty or "district" not in stop_df.columns or metric not in stop_df.columns:
        return pd.DataFrame()
    return stop_df.groupby("district", as_index=False)[metric].sum().sort_values(metric, ascending=False)


def hourly_totals(hourly_df: pd.DataFrame, metric: str = "boardings") -> pd.DataFrame:
    """시간대별 지표 합계를 계산합니다."""
    if hourly_df.empty or "hour" not in hourly_df.columns or metric not in hourly_df.columns:
        return pd.DataFrame()
    return hourly_df.groupby("hour", as_index=False)[metric].sum().sort_values("hour")


def interpret_correlation(value: float) -> str:
    """상관계수 크기를 약함, 보통, 강함으로 해석합니다."""
    if pd.isna(value):
        return "계산 불가"
    strength = abs(value)
    if strength < 0.2:
        return "약한"
    if strength < 0.4:
        return "다소 약한"
    if strength < 0.6:
        return "보통 수준의"
    if strength < 0.8:
        return "강한"
    return "매우 강한"


def correlation_analysis(stop_df: pd.DataFrame, x: str = "route_count", y: str = "boardings") -> dict:
    """경유 노선 수와 승차 인원의 Pearson, Spearman 상관계수를 계산하고 문장으로 해석합니다."""
    if stop_df.empty or x not in stop_df.columns or y not in stop_df.columns:
        return {"pearson": np.nan, "spearman": np.nan, "pearson_sentence": "상관관계를 계산할 데이터가 부족합니다."}
    data = stop_df[[x, y]].dropna()
    data = data[(data[x] > 0) & (data[y] >= 0)]
    if len(data) < 3 or data[x].nunique() < 2 or data[y].nunique() < 2:
        return {"pearson": np.nan, "spearman": np.nan, "pearson_sentence": "상관관계를 계산할 데이터가 부족합니다."}
    pearson = data[x].corr(data[y], method="pearson")
    spearman = data[x].corr(data[y], method="spearman")
    pearson_strength = interpret_correlation(pearson)
    spearman_strength = interpret_correlation(spearman)
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pearson_sentence": f"경유 노선 수와 승차 인원의 Pearson 상관계수는 {pearson:.2f}로 나타났습니다. 이는 두 변수 사이의 선형관계가 {pearson_strength} 관계임을 의미합니다.",
        "spearman_sentence": f"Spearman 상관계수는 {spearman:.2f}로 나타났습니다. 이는 순위 기준 관계가 {spearman_strength} 관계임을 의미하며, 인과관계를 뜻하지는 않습니다.",
    }


def summary_statistics(stop_df: pd.DataFrame, metric: str = "boardings") -> pd.DataFrame:
    """평균, 중앙값, 표준편차, 사분위수, 변동계수 등 기본 통계를 계산합니다."""
    if stop_df.empty or metric not in stop_df.columns:
        return pd.DataFrame()
    series = stop_df[metric].dropna()
    if series.empty:
        return pd.DataFrame()
    mean = series.mean()
    return pd.DataFrame(
        [
            {
                "지표": metric,
                "평균": mean,
                "중앙값": series.median(),
                "표준편차": series.std(),
                "최솟값": series.min(),
                "1사분위수": series.quantile(0.25),
                "3사분위수": series.quantile(0.75),
                "최댓값": series.max(),
                "변동계수": series.std() / mean if mean else np.nan,
            }
        ]
    )


def imbalance_candidates(
    stop_df: pd.DataFrame,
    demand_quantile: float = 0.75,
    route_quantile: float = 0.50,
    per_route_quantile: float = 0.90,
) -> tuple[pd.DataFrame, dict]:
    """수요는 높고 경유 노선 수는 적은 추가 검토 후보 정류소를 찾습니다."""
    required = {"boardings", "route_count", "boardings_per_route"}
    if stop_df.empty or not required.issubset(stop_df.columns):
        return pd.DataFrame(), {"demand_threshold": np.nan, "route_threshold": np.nan, "per_route_threshold": np.nan}

    data = stop_df.dropna(subset=list(required)).copy()
    data = data[(data["route_count"] > 0) & (data["boardings"] >= 0)]
    if data.empty:
        return pd.DataFrame(), {"demand_threshold": np.nan, "route_threshold": np.nan, "per_route_threshold": np.nan}

    thresholds = {
        "demand_threshold": data["boardings"].quantile(demand_quantile),
        "route_threshold": data["route_count"].quantile(route_quantile),
        "per_route_threshold": data["boardings_per_route"].quantile(per_route_quantile),
    }
    candidates = data[
        (data["boardings"] >= thresholds["demand_threshold"])
        & (data["route_count"] <= thresholds["route_threshold"])
        & (data["boardings_per_route"] >= thresholds["per_route_threshold"])
    ].sort_values("boardings_per_route", ascending=False)
    return candidates, thresholds


def supply_mismatch_tables(stop_df: pd.DataFrame, n: int = 10) -> dict[str, pd.DataFrame]:
    """노선 수가 적지만 이용객이 많은 정류소와 반대 사례를 계산합니다."""
    if stop_df.empty or not {"boardings", "route_count"}.issubset(stop_df.columns):
        return {}
    data = stop_df.dropna(subset=["boardings", "route_count"]).copy()
    if data.empty:
        return {}
    high_demand = data["boardings"] >= data["boardings"].quantile(0.75)
    low_routes = data["route_count"] <= data["route_count"].quantile(0.50)
    high_routes = data["route_count"] >= data["route_count"].quantile(0.75)
    low_demand = data["boardings"] <= data["boardings"].quantile(0.50)
    return {
        "노선 수는 적지만 이용객이 많은 정류소": data[high_demand & low_routes].sort_values("boardings", ascending=False).head(n),
        "노선 수는 많지만 이용객이 적은 정류소": data[high_routes & low_demand].sort_values("route_count", ascending=False).head(n),
    }


def outlier_stops(stop_df: pd.DataFrame, metric: str = "boardings") -> pd.DataFrame:
    """IQR 기준으로 이상치 가능성이 있는 정류소를 찾습니다."""
    if stop_df.empty or metric not in stop_df.columns:
        return pd.DataFrame()
    series = stop_df[metric].dropna()
    if series.empty:
        return pd.DataFrame()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    return stop_df[stop_df[metric] > upper].sort_values(metric, ascending=False)


def monthly_yoy_growth(monthly_df: pd.DataFrame, metric: str = "boardings") -> pd.DataFrame:
    """월별 데이터가 있을 때 전년 대비 증감률을 계산합니다."""
    required = {"_merge_key", "year", "month", metric}
    if monthly_df.empty or not required.issubset(monthly_df.columns):
        return pd.DataFrame()
    base_cols = [col for col in ["_merge_key", "stop_name", "district", "year", "month", metric] if col in monthly_df.columns]
    base = monthly_df[base_cols].dropna(subset=["year", "month", metric]).copy()
    previous = base[["_merge_key", "year", "month", metric]].copy()
    previous["year"] = previous["year"] + 1
    previous = previous.rename(columns={metric: "previous_year_value"})
    merged = base.merge(previous, on=["_merge_key", "year", "month"], how="left")
    merged = merged[merged["previous_year_value"].notna() & (merged["previous_year_value"] != 0)].copy()
    if merged.empty:
        return pd.DataFrame()
    merged["yoy_growth_rate"] = (merged[metric] - merged["previous_year_value"]) / merged["previous_year_value"] * 100
    return merged


def create_analysis_tables(stop_df: pd.DataFrame, hourly_df: pd.DataFrame, monthly_df: pd.DataFrame, top_n: int = 10) -> dict:
    """핵심 분석 결과 표를 한 번에 만듭니다."""
    tables = {
        f"전체 승차 인원이 많은 정류소 TOP {top_n}": top_stops(stop_df, "boardings", top_n),
        f"노선당 승차 인원이 높은 정류소 TOP {top_n}": top_stops(stop_df, "boardings_per_route", top_n),
        "구·군별 전체 승차 인원": district_totals(stop_df, "boardings"),
        "시간대별 전체 승차 인원": hourly_totals(hourly_df, "boardings"),
        f"출근 시간 이용객 TOP {top_n}": top_stops(stop_df, "morning_boardings", top_n),
        f"퇴근 시간 이용객 TOP {top_n}": top_stops(stop_df, "evening_boardings", top_n),
        "정류소 유형별 개수": stop_df["stop_type"].value_counts().reset_index(name="count").rename(columns={"index": "stop_type"}) if "stop_type" in stop_df.columns else pd.DataFrame(),
        "승차와 하차의 차이가 큰 정류소": top_stops(stop_df.assign(abs_diff=stop_df.get("boarding_alighting_diff", pd.Series(dtype=float)).abs()), "abs_diff", top_n) if "boarding_alighting_diff" in stop_df.columns else pd.DataFrame(),
        "구·군별 노선당 승차 인원 평균": stop_df.groupby("district", as_index=False)["boardings_per_route"].mean() if {"district", "boardings_per_route"}.issubset(stop_df.columns) else pd.DataFrame(),
        "이상치 정류소": outlier_stops(stop_df, "boardings"),
    }
    tables.update(supply_mismatch_tables(stop_df, top_n))
    yoy = monthly_yoy_growth(monthly_df)
    if not yoy.empty:
        tables[f"이용량 증가율 TOP {top_n}"] = yoy.sort_values("yoy_growth_rate", ascending=False).head(top_n)
        tables[f"이용량 감소율 TOP {top_n}"] = yoy.sort_values("yoy_growth_rate").head(top_n)
    return tables


def kmeans_cluster_hourly_patterns(hourly_df: pd.DataFrame, min_samples: int = 30) -> dict:
    """데이터가 충분할 때 시간대별 이용 비율을 K-means로 군집화합니다."""
    if hourly_df.empty or not {"_merge_key", "hour", "boardings"}.issubset(hourly_df.columns):
        return {"assignments": pd.DataFrame(), "patterns": pd.DataFrame(), "message": "K-means를 실행할 시간대별 승차 데이터가 없습니다."}
    pivot = hourly_df.pivot_table(index="_merge_key", columns="hour", values="boardings", aggfunc="sum").fillna(0)
    pivot = pivot[pivot.sum(axis=1) > 0]
    if len(pivot) < min_samples:
        return {"assignments": pd.DataFrame(), "patterns": pd.DataFrame(), "message": "K-means를 실행하기에는 정류소 수가 부족합니다."}

    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return {"assignments": pd.DataFrame(), "patterns": pd.DataFrame(), "message": "scikit-learn이 설치되어 있지 않아 K-means를 건너뜁니다."}

    ratios = pivot.div(pivot.sum(axis=1), axis=0)
    scaled = StandardScaler().fit_transform(ratios)
    max_k = min(6, len(ratios) - 1)
    best_k = 2
    best_score = -1
    for k in range(2, max_k + 1):
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        score = silhouette_score(scaled, labels)
        if score > best_score:
            best_k = k
            best_score = score

    labels = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit_predict(scaled)
    assignments = pd.DataFrame({"_merge_key": ratios.index, "cluster": labels})
    patterns = ratios.assign(cluster=labels).groupby("cluster").mean().reset_index()
    patterns["cluster_name"] = patterns.apply(interpret_cluster_pattern, axis=1)
    return {"assignments": assignments, "patterns": patterns, "message": f"K-means 군집 수는 실루엣 점수 기준 {best_k}개로 선택했습니다."}


def interpret_cluster_pattern(row: pd.Series) -> str:
    """군집별 평균 시간대 패턴을 사람이 이해하기 쉬운 이름으로 바꿉니다."""
    hour_cols = [col for col in row.index if isinstance(col, (int, float))]
    if not hour_cols:
        return "해석 불가"
    morning = row[[hour for hour in hour_cols if 7 <= int(hour) <= 9]].sum()
    evening = row[[hour for hour in hour_cols if 17 <= int(hour) <= 20]].sum()
    peak_hour = int(row[hour_cols].idxmax())
    if morning >= 0.25 and evening >= 0.25:
        return "출퇴근 집중형"
    if morning >= 0.25 and morning > evening:
        return "출근 집중형"
    if evening >= 0.25 and evening > morning:
        return "퇴근 집중형"
    return f"{peak_hour}시 생활형"


def generate_analysis_summary_markdown(
    stop_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    candidates: pd.DataFrame,
    correlation: dict,
) -> str:
    """발표 준비용 분석 결과 요약 문서를 만듭니다."""
    if stop_df.empty:
        return "# 분석 결과 요약\n\n분석 가능한 정류소 데이터가 없습니다.\n"

    def top_name(df: pd.DataFrame, metric: str) -> str:
        """특정 지표의 1위 정류소명을 찾습니다."""
        if df.empty or metric not in df.columns or "stop_name" not in df.columns:
            return "계산 불가"
        valid = df.dropna(subset=[metric])
        if valid.empty:
            return "계산 불가"
        row = valid.sort_values(metric, ascending=False).iloc[0]
        return f"{row.get('stop_name', '이름 없음')} ({row[metric]:,.0f})"

    peak_hour = "계산 불가"
    if not hourly_df.empty and {"hour", "boardings"}.issubset(hourly_df.columns):
        hourly_total = hourly_df.groupby("hour")["boardings"].sum()
        if not hourly_total.empty:
            peak_hour = f"{int(hourly_total.idxmax())}시 ({hourly_total.max():,.0f})"

    morning_case = top_name(stop_df[stop_df.get("stop_type", "") == "출근형"] if "stop_type" in stop_df.columns else pd.DataFrame(), "boardings")
    evening_case = top_name(stop_df[stop_df.get("stop_type", "") == "퇴근형"] if "stop_type" in stop_df.columns else pd.DataFrame(), "boardings")
    district_summary = "계산 불가"
    if {"district", "boardings"}.issubset(stop_df.columns):
        district_top = stop_df.groupby("district")["boardings"].sum().sort_values(ascending=False)
        if not district_top.empty:
            district_summary = ", ".join([f"{idx}: {value:,.0f}" for idx, value in district_top.head(5).items()])

    candidate_text = "후보 없음"
    if not candidates.empty:
        candidate_text = ", ".join(candidates.get("stop_name", pd.Series(dtype=str)).dropna().astype(str).head(10))

    return f"""# 분석 결과 요약

- 가장 이용객이 많은 정류소: {top_name(stop_df, "boardings")}
- 가장 혼잡한 시간대: {peak_hour}
- 출근형 정류소 대표 사례: {morning_case}
- 퇴근형 정류소 대표 사례: {evening_case}
- 노선당 승차 인원이 높은 정류소: {top_name(stop_df, "boardings_per_route")}
- 수요·공급 불균형 추가 검토 후보: {candidate_text}
- 구·군별 주요 특징: {district_summary}
- 분석 결과의 한계: 하차 태그 누락, 현금 승차 제외 가능성, 배차 간격과 차량 크기 부재, 주변 시설 정보 부재로 인해 노선 부족을 확정할 수 없습니다.
- 정책적 활용 가능성: 후보 정류소를 대상으로 배차 간격, 환승 수요, 주변 시설, 실제 혼잡도 조사를 결합하면 노선 조정 또는 현장 점검 우선순위 설정에 활용할 수 있습니다.

## 상관관계 해석

{correlation.get("pearson_sentence", "상관관계를 계산할 수 없습니다.")}

상관관계는 두 변수가 함께 움직이는 정도를 보여줄 뿐이며 인과관계를 의미하지 않습니다.
"""
