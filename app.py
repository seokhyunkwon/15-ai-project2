from pathlib import Path
from html import escape

import pandas as pd
import streamlit as st

from src.analysis import (
    correlation_analysis,
    create_analysis_tables,
    generate_analysis_summary_markdown,
    imbalance_candidates,
    kmeans_cluster_hourly_patterns,
    monthly_yoy_growth,
)
from src.data_loader import audit_to_dataframe
from src.preprocessing import normalize_stop_id
from src.utils import (
    ensure_directories,
    format_number,
    make_download_csv,
    save_dataframe_csv,
    set_korean_font,
)
from src.visualization import (
    create_pydeck_map,
    metric_label as plot_metric_label,
    plot_board_alight_line,
    plot_district_bar,
    plot_district_hour_heatmap,
    plot_hourly_line,
    plot_stop_comparison_line,
    plot_stop_hour_heatmap,
    plot_supply_demand_scatter,
    plot_season_bus_box,
    plot_top_stops,
    plot_type_pie,
    plot_weather_bus_dual_line,
    plot_weather_correlation_heatmap,
    plot_weather_monthly_line,
    plot_weather_scatter,
    save_static_figures,
)
from src.weather_analysis import (
    aggregate_bus_monthly,
    available_weather_variables,
    calculate_weather_correlations,
    infer_bus_time_unit,
    merge_bus_weather_monthly,
    selected_weather_correlation_summary,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
PROCESSED_DIR = OUTPUT_DIR / "processed"
FIGURE_DIR = OUTPUT_DIR / "figures"

PROCESSED_CACHE_FILES = {
    "stop_summary": "stop_summary.csv",
    "hourly_summary": "hourly_summary.csv",
    "monthly_summary": "monthly_summary.csv",
    "column_mapping": "column_mapping.csv",
    "weather_daily": "weather_daily.csv",
    "weather_monthly": "weather_monthly.csv",
    "weather_mapping": "weather_mapping.csv",
    "weather_audit": "weather_audit.csv",
    "data_check_report": "data_check_report.csv",
}


# 정류소 유형 분류 기준입니다. 발표 목적이나 데이터 특성에 맞게 쉽게 바꿀 수 있습니다.
MORNING_START_HOUR = 7
MORNING_END_HOUR = 9
EVENING_START_HOUR = 17
EVENING_END_HOUR = 20
HIGH_CONCENTRATION_THRESHOLD = 25.0
LOW_USAGE_QUANTILE = 0.25
YOY_MIN_PREVIOUS_VALUE = 1_000
YOY_MIN_CURRENT_VALUE_FOR_DECREASE = 1_000
YOY_BASE_YEAR = 2025
YOY_TARGET_YEAR = 2026


def empty_message(message: str) -> None:
    """빈 데이터로 그래프를 그리지 않도록 안내문을 표시합니다."""
    st.info(message)


def data_fingerprint(data_dir: Path) -> tuple:
    """CSV 파일이 바뀌면 Streamlit 캐시가 갱신되도록 파일 상태를 요약합니다."""
    if not data_dir.exists():
        return tuple()
    fingerprint = []
    for path in sorted(data_dir.rglob("*.csv")):
        stat = path.stat()
        fingerprint.append((str(path.relative_to(data_dir)), int(stat.st_mtime), stat.st_size))
    return tuple(fingerprint)


def processed_cache_is_fresh(data_dir: Path, processed_dir: Path) -> bool:
    """전처리 CSV가 원본 CSV보다 최신이면 빠른 로딩에 사용할 수 있는지 확인합니다."""
    required = ["stop_summary", "hourly_summary", "monthly_summary", "weather_monthly", "data_check_report"]
    paths = [processed_dir / PROCESSED_CACHE_FILES[name] for name in required]
    if not all(path.exists() for path in paths):
        return False
    raw_csv_files = list(data_dir.rglob("*.csv")) if data_dir.exists() else []
    if not raw_csv_files:
        return False
    latest_raw = max(path.stat().st_mtime for path in raw_csv_files)
    oldest_processed = min(path.stat().st_mtime for path in paths)
    return oldest_processed >= latest_raw


def processed_cache_exists(processed_dir: Path) -> bool:
    """Streamlit 앱이 바로 읽을 정제 CSV가 있는지 확인합니다."""
    required = ["stop_summary", "hourly_summary", "monthly_summary", "weather_monthly", "data_check_report"]
    return all((processed_dir / PROCESSED_CACHE_FILES[name]).exists() for name in required)


def load_processed_cache(processed_dir: Path) -> dict:
    """이미 생성된 전처리 CSV를 읽어 앱 시작 시간을 줄입니다."""
    bundle = {}
    for key, file_name in PROCESSED_CACHE_FILES.items():
        path = processed_dir / file_name
        if path.exists():
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
            if "stop_id" in df.columns:
                df["stop_id"] = normalize_stop_id(df["stop_id"])
            bundle[key] = df
    audit_df = bundle.get("data_check_report", pd.DataFrame())
    if not audit_df.empty:
        bundle["audit"] = [
            {
                "file_name": row.get("파일명"),
                "rows": row.get("행 수"),
                "columns_count": row.get("열 수"),
                "columns": str(row.get("컬럼명", "")).split(", ") if pd.notna(row.get("컬럼명")) else [],
                "duplicate_rows": row.get("중복 데이터 개수"),
                "encoding": row.get("문자 인코딩"),
                "read_error": row.get("읽기 오류", ""),
                "head": [],
                "dtypes": {},
                "missing_values": {},
            }
            for _, row in audit_df.iterrows()
        ]
    else:
        bundle["audit"] = []
    bundle["loaded_from_processed_cache"] = True
    return bundle


@st.cache_data(show_spinner="정제된 분석 CSV를 불러오는 중입니다.")
def load_project_data(data_dir: str, fingerprint: tuple) -> dict:
    """Streamlit에서는 원본 CSV가 아니라 정제된 CSV만 불러옵니다."""
    data_path = Path(data_dir)
    del fingerprint
    if not processed_cache_exists(PROCESSED_DIR):
        return {
            "audit": [],
            "stop_summary": pd.DataFrame(),
            "hourly_summary": pd.DataFrame(),
            "monthly_summary": pd.DataFrame(),
            "processed_cache_missing": True,
        }

    bundle = load_processed_cache(PROCESSED_DIR)
    bundle["processed_cache_stale"] = not processed_cache_is_fresh(data_path, PROCESSED_DIR)
    return bundle


def write_processed_outputs(bundle: dict, candidates: pd.DataFrame) -> None:
    """전처리 데이터, 분석 결과 CSV, 데이터 점검 결과, 요약 문서를 저장합니다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = audit_to_dataframe(bundle.get("audit", []))
    save_dataframe_csv(audit_df, PROCESSED_DIR / "data_check_report.csv")

    for name in [
        "stop_summary",
        "hourly_summary",
        "monthly_summary",
        "column_mapping",
        "weather_daily",
        "weather_monthly",
        "weather_mapping",
        "weather_audit",
    ]:
        df = bundle.get(name, pd.DataFrame())
        if isinstance(df, pd.DataFrame):
            save_dataframe_csv(df, PROCESSED_DIR / f"{name}.csv")

    save_dataframe_csv(candidates, PROCESSED_DIR / "imbalance_candidates.csv")

    correlation = correlation_analysis(bundle.get("stop_summary", pd.DataFrame()))
    summary_text = generate_analysis_summary_markdown(
        bundle.get("stop_summary", pd.DataFrame()),
        bundle.get("hourly_summary", pd.DataFrame()),
        bundle.get("monthly_summary", pd.DataFrame()),
        candidates,
        correlation,
    )
    (PROCESSED_DIR / "analysis_summary.md").write_text(summary_text, encoding="utf-8")


def save_outputs_on_request(bundle: dict, stop_df: pd.DataFrame, hourly_df: pd.DataFrame) -> None:
    """사용자가 요청한 현재 필터 결과를 별도 CSV로 저장합니다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    candidates, _ = imbalance_candidates(stop_df)
    save_dataframe_csv(stop_df, PROCESSED_DIR / "current_filtered_stop_summary.csv")
    save_dataframe_csv(hourly_df, PROCESSED_DIR / "current_filtered_hourly_summary.csv")
    save_dataframe_csv(candidates, PROCESSED_DIR / "current_filtered_imbalance_candidates.csv")

    correlation = correlation_analysis(stop_df)
    summary_text = generate_analysis_summary_markdown(
        stop_df,
        hourly_df,
        bundle.get("monthly_summary", pd.DataFrame()),
        candidates,
        correlation,
    )
    (PROCESSED_DIR / "current_filtered_analysis_summary.md").write_text(summary_text, encoding="utf-8")


def filter_stop_data(stop_df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """사이드바 조건을 정류소 요약 데이터에 적용합니다."""
    filtered = stop_df.copy()
    if filtered.empty:
        return filtered

    if filters["districts"] and "district" in filtered.columns:
        filtered = filtered[filtered["district"].isin(filters["districts"])]
    if filters["stops"] and "stop_name" in filtered.columns:
        filtered = filtered[filtered["stop_name"].isin(filters["stops"])]
    if filters["types"] and "stop_type" in filtered.columns:
        filtered = filtered[filtered["stop_type"].isin(filters["types"])]
    if "boardings" in filtered.columns:
        filtered = filtered[filtered["boardings"].fillna(0) >= filters["min_boardings"]]
    if "route_count" in filtered.columns and filters["max_route_count"] is not None:
        filtered = filtered[filtered["route_count"].fillna(10**12) <= filters["max_route_count"]]
    if "boardings_per_route" in filtered.columns:
        filtered = filtered[
            filtered["boardings_per_route"].fillna(-1) >= filters["min_boardings_per_route"]
        ]
    return filtered


def filter_hourly_data(hourly_df: pd.DataFrame, stop_df: pd.DataFrame, hours: tuple[int, int]) -> pd.DataFrame:
    """정류소 필터와 시간대 필터를 시간대별 데이터에 적용합니다."""
    if hourly_df.empty or stop_df.empty or "_merge_key" not in hourly_df.columns:
        return pd.DataFrame()
    valid_keys = set(stop_df["_merge_key"].dropna())
    filtered = hourly_df[hourly_df["_merge_key"].isin(valid_keys)].copy()
    if "hour" in filtered.columns:
        filtered = filtered[(filtered["hour"] >= hours[0]) & (filtered["hour"] <= hours[1])]
    return filtered


def metric_choice_to_column(choice: str) -> str:
    """화면의 한글 지표명을 실제 데이터 컬럼명으로 바꿉니다."""
    mapping = {
        "승차 인원": "boardings",
        "하차 인원": "alightings",
        "전체 이용객": "total_riders",
    }
    return mapping.get(choice, "boardings")


def show_metric(label: str, value) -> None:
    """숫자 지표를 천 단위 쉼표와 함께 표시합니다."""
    st.metric(label, format_number(value))


def render_plotly_chart(fig, empty_text: str, caption: bool = False) -> None:
    """그래프 객체가 있을 때만 표시하고 Streamlit 내부 객체가 화면에 노출되지 않게 합니다."""
    if fig is not None:
        st.plotly_chart(fig, use_container_width=True)
    elif caption:
        st.caption(empty_text)
    else:
        empty_message(empty_text)


def sidebar_filters(stop_df: pd.DataFrame, monthly_df: pd.DataFrame) -> dict:
    """Streamlit 사이드바 필터 값을 모아서 반환합니다."""
    st.sidebar.header("필터")
    districts = []
    stops = []
    stop_types = []

    if not stop_df.empty and "district" in stop_df.columns:
        district_options = sorted(stop_df["district"].dropna().astype(str).unique())
        districts = st.sidebar.multiselect("구·군 선택", district_options)

    if not stop_df.empty and "stop_name" in stop_df.columns:
        stop_options = sorted(stop_df["stop_name"].dropna().astype(str).unique())
        stops = st.sidebar.multiselect("정류소 선택", stop_options)

    if not monthly_df.empty and {"year", "month"}.issubset(monthly_df.columns):
        period_options = (
            monthly_df[["year", "month"]]
            .dropna()
            .drop_duplicates()
            .sort_values(["year", "month"])
            .assign(period=lambda x: x["year"].astype(int).astype(str) + "-" + x["month"].astype(int).astype(str).str.zfill(2))
        )
        periods = st.sidebar.multiselect("연도·월 선택", period_options["period"].tolist())
    else:
        periods = []
        st.sidebar.caption("날짜 또는 연도·월 컬럼이 있으면 기간 필터가 표시됩니다.")

    hour_range = st.sidebar.slider("시간대 선택", 0, 23, (0, 23))

    if not stop_df.empty and "stop_type" in stop_df.columns:
        type_options = sorted(stop_df["stop_type"].dropna().astype(str).unique())
        stop_types = st.sidebar.multiselect("정류소 유형 선택", type_options)

    metric_label = st.sidebar.selectbox("승차·하차·전체 이용객 선택", ["승차 인원", "하차 인원", "전체 이용객"])
    top_n = st.sidebar.slider("TOP N 선택", 5, 30, 10)
    min_boardings = st.sidebar.number_input("최소 승차 인원", min_value=0, value=0, step=100)

    max_route_count = None
    if not stop_df.empty and "route_count" in stop_df.columns and stop_df["route_count"].notna().any():
        max_routes = int(max(1, stop_df["route_count"].max()))
        max_route_count = st.sidebar.number_input("최대 경유 노선 수", min_value=0, value=max_routes, step=1)
    else:
        st.sidebar.caption("경유 노선 수 컬럼이 있으면 최대 노선 수 필터가 표시됩니다.")

    min_boardings_per_route = st.sidebar.number_input("노선당 승차 인원 기준", min_value=0.0, value=0.0, step=100.0)
    show_map = st.sidebar.checkbox("지도 표시 여부", value=True)
    show_static = st.sidebar.checkbox("정적 그래프 표시 여부", value=False)

    return {
        "districts": districts,
        "stops": stops,
        "periods": periods,
        "hours": hour_range,
        "types": stop_types,
        "metric_label": metric_label,
        "metric_col": metric_choice_to_column(metric_label),
        "top_n": top_n,
        "min_boardings": min_boardings,
        "max_route_count": max_route_count,
        "min_boardings_per_route": min_boardings_per_route,
        "show_map": show_map,
        "show_static": show_static,
    }


def overview_tab(stop_df: pd.DataFrame, hourly_df: pd.DataFrame, top_n: int, metric_col: str) -> None:
    """전체 현황 탭을 구성합니다."""
    if stop_df.empty:
        empty_message("필터 조건에 맞는 정류소 데이터가 없습니다.")
        return

    metric_cols = st.columns(6)
    with metric_cols[0]:
        show_metric("전체 승차 인원", stop_df.get("boardings", pd.Series(dtype=float)).sum())
    with metric_cols[1]:
        show_metric("전체 하차 인원", stop_df.get("alightings", pd.Series(dtype=float)).sum())
    with metric_cols[2]:
        show_metric("분석 대상 정류소 수", len(stop_df))
    with metric_cols[3]:
        top_stop = "-"
        if "boardings" in stop_df.columns and stop_df["boardings"].notna().any():
            top_stop = stop_df.sort_values("boardings", ascending=False).iloc[0].get("stop_name", "-")
        st.metric("가장 이용객이 많은 정류소", top_stop)
    with metric_cols[4]:
        peak_hour = "-"
        if not hourly_df.empty and "hour" in hourly_df.columns and "boardings" in hourly_df.columns:
            hourly_total = hourly_df.groupby("hour", as_index=False)["boardings"].sum()
            if not hourly_total.empty:
                peak_hour = f"{int(hourly_total.sort_values('boardings', ascending=False).iloc[0]['hour'])}시"
        st.metric("가장 혼잡한 시간대", peak_hour)
    with metric_cols[5]:
        per_route_stop = "-"
        if "boardings_per_route" in stop_df.columns and stop_df["boardings_per_route"].notna().any():
            per_route_stop = stop_df.sort_values("boardings_per_route", ascending=False).iloc[0].get("stop_name", "-")
        st.metric("노선당 승차 인원 최고", per_route_stop)

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_top_stops(stop_df, metric=metric_col, n=top_n, title=f"정류소별 {plot_metric_label(metric_col)} TOP {top_n}")
        render_plotly_chart(fig, "정류소 순위 그래프를 그릴 수 없습니다.")
    with col2:
        fig = plot_district_bar(stop_df, metric=metric_col, title="구·군별 승차 인원")
        render_plotly_chart(fig, "구·군 컬럼이 없어 구·군별 그래프를 표시할 수 없습니다.")

    col3, col4 = st.columns(2)
    with col3:
        fig = plot_hourly_line(hourly_df, metric=metric_col, title="시간대별 승차 인원")
        render_plotly_chart(fig, "시간대 컬럼이 없어 시간대별 그래프를 표시할 수 없습니다.")
    with col4:
        fig = plot_type_pie(stop_df)
        render_plotly_chart(fig, "정류소 유형을 계산할 수 없습니다.")


def hourly_tab(stop_df: pd.DataFrame, hourly_df: pd.DataFrame, top_n: int, metric_col: str) -> None:
    """시간대별 분석 탭을 구성합니다."""
    if hourly_df.empty:
        empty_message("시간대 컬럼이 있는 데이터가 없어 시간대별 분석을 표시할 수 없습니다.")
        return

    col1, col2 = st.columns(2)
    with col1:
        fig = plot_hourly_line(hourly_df, metric=metric_col, title="시간대별 전체 승차 인원")
        render_plotly_chart(fig, "시간대별 전체 승차 인원을 표시할 수 없습니다.")
    with col2:
        fig = plot_board_alight_line(hourly_df)
        render_plotly_chart(fig, "승차·하차 비교에 필요한 컬럼이 부족합니다.")

    col3, col4 = st.columns(2)
    with col3:
        fig = plot_stop_hour_heatmap(hourly_df, stop_df, metric=metric_col, top_n=20)
        render_plotly_chart(fig, "정류소 × 시간대 히트맵을 표시할 수 없습니다.")
    with col4:
        fig = plot_district_hour_heatmap(hourly_df, metric=metric_col)
        render_plotly_chart(fig, "구·군 × 시간대 히트맵을 표시할 수 없습니다.")

    morning = stop_df.sort_values("morning_boardings", ascending=False).head(top_n) if "morning_boardings" in stop_df.columns else pd.DataFrame()
    evening = stop_df.sort_values("evening_boardings", ascending=False).head(top_n) if "evening_boardings" in stop_df.columns else pd.DataFrame()
    col5, col6 = st.columns(2)
    with col5:
        fig = plot_top_stops(morning, metric="morning_boardings", n=top_n, title=f"출근 시간 이용객 TOP {top_n}")
        render_plotly_chart(fig, "출근 시간 이용객을 계산할 수 없습니다.")
    with col6:
        fig = plot_top_stops(evening, metric="evening_boardings", n=top_n, title=f"퇴근 시간 이용객 TOP {top_n}")
        render_plotly_chart(fig, "퇴근 시간 이용객을 계산할 수 없습니다.")

    if "stop_name" in stop_df.columns:
        selected = st.multiselect(
            "시간대 패턴을 비교할 정류소",
            sorted(stop_df["stop_name"].dropna().astype(str).unique()),
            max_selections=5,
        )
        fig = plot_stop_comparison_line(hourly_df, selected, metric=metric_col)
        render_plotly_chart(fig, "정류소를 선택하면 시간대 패턴 비교 그래프가 표시됩니다.", caption=True)


def stop_detail_tab(stop_df: pd.DataFrame, hourly_df: pd.DataFrame, monthly_df: pd.DataFrame) -> None:
    """정류소별 상세 분석 탭을 구성합니다."""
    if stop_df.empty or "stop_name" not in stop_df.columns:
        empty_message("정류소명 컬럼이 없어 정류소별 상세 분석을 표시할 수 없습니다.")
        return

    stop_name = st.selectbox("상세 분석 정류소 선택", sorted(stop_df["stop_name"].dropna().astype(str).unique()))
    selected = stop_df[stop_df["stop_name"].astype(str) == stop_name]
    if selected.empty:
        empty_message("선택한 정류소 데이터가 없습니다.")
        return

    row = selected.iloc[0]
    cols = st.columns(5)
    with cols[0]:
        st.metric("정류소명", row.get("stop_name", "-"))
    with cols[1]:
        st.metric("행정구역", row.get("district", "-"))
    with cols[2]:
        show_metric("경유 노선 수", row.get("route_count"))
    with cols[3]:
        show_metric("전체 승차 인원", row.get("boardings"))
    with cols[4]:
        show_metric("전체 하차 인원", row.get("alightings"))

    cols2 = st.columns(5)
    with cols2[0]:
        show_metric("노선당 승차 인원", row.get("boardings_per_route"))
    with cols2[1]:
        st.metric("최대 혼잡 시간대", row.get("peak_hour_label", "-"))
    with cols2[2]:
        show_metric("출근 시간 집중도(%)", row.get("morning_concentration"))
    with cols2[3]:
        show_metric("퇴근 시간 집중도(%)", row.get("evening_concentration"))
    with cols2[4]:
        st.metric("정류소 유형", row.get("stop_type", "-"))

    fig = plot_stop_comparison_line(hourly_df, [stop_name], metric="boardings", include_average=True)
    render_plotly_chart(fig, "선택 정류소의 시간대별 그래프를 표시할 수 없습니다.")

    if not monthly_df.empty and "_merge_key" in monthly_df.columns:
        selected_keys = selected["_merge_key"].dropna().unique()
        stop_monthly = monthly_df[monthly_df["_merge_key"].isin(selected_keys)].copy()
        if not stop_monthly.empty and {"year", "month"}.issubset(stop_monthly.columns):
            stop_monthly["period"] = stop_monthly["year"].astype(int).astype(str) + "-" + stop_monthly["month"].astype(int).astype(str).str.zfill(2)
            metric = "boardings" if "boardings" in stop_monthly.columns else "passengers"
            st.line_chart(stop_monthly.set_index("period")[metric])


def imbalance_tab(stop_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """수요·공급 불균형 분석 탭을 구성하고 후보 정류소를 반환합니다."""
    if stop_df.empty:
        empty_message("수요·공급 불균형 분석에 사용할 데이터가 없습니다.")
        return pd.DataFrame()

    st.caption("이 분석은 실제 노선 부족을 확정하지 않고, 추가 검토가 필요한 후보 정류소를 찾는 용도입니다.")
    col1, col2, col3 = st.columns(3)
    with col1:
        demand_q = st.slider("전체 승차 인원 상위 기준", 0.50, 0.95, 0.75, 0.05)
    with col2:
        route_q = st.slider("경유 노선 수 하위 기준", 0.10, 0.90, 0.50, 0.05)
    with col3:
        per_route_q = st.slider("노선당 승차 인원 상위 기준", 0.50, 0.99, 0.90, 0.01)

    candidates, thresholds = imbalance_candidates(
        stop_df,
        demand_quantile=demand_q,
        route_quantile=route_q,
        per_route_quantile=per_route_q,
    )
    st.write(
        f"후보 기준: 전체 승차 인원 {format_number(thresholds.get('demand_threshold'))} 이상, "
        f"경유 노선 수 {format_number(thresholds.get('route_threshold'))} 이하, "
        f"노선당 승차 인원 {format_number(thresholds.get('per_route_threshold'))} 이상"
    )

    fig = plot_supply_demand_scatter(stop_df, thresholds)
    render_plotly_chart(fig, "산점도에 필요한 승차 인원 또는 경유 노선 수 컬럼이 부족합니다.")

    col4, col5 = st.columns(2)
    with col4:
        fig = plot_top_stops(stop_df, metric="boardings_per_route", n=top_n, title=f"노선당 승차 인원 TOP {top_n}")
        render_plotly_chart(fig, "노선당 승차 인원을 계산할 수 없습니다.")
    with col5:
        correlation = correlation_analysis(stop_df)
        st.subheader("상관관계 해석")
        st.write(correlation.get("pearson_sentence", "상관관계를 계산할 수 없습니다."))
        if correlation.get("spearman_sentence"):
            st.write(correlation["spearman_sentence"])

    display_cols = [
        "stop_name",
        "district",
        "boardings",
        "alightings",
        "total_riders",
        "route_count",
        "boardings_per_route",
        "morning_concentration",
        "evening_concentration",
        "peak_hour_label",
        "stop_type",
    ]
    display_cols = [col for col in display_cols if col in candidates.columns]
    st.dataframe(candidates[display_cols], use_container_width=True)
    st.download_button(
        "추가 검토 후보 정류소 CSV 다운로드",
        data=make_download_csv(candidates[display_cols] if display_cols else candidates),
        file_name="imbalance_candidates.csv",
        mime="text/csv",
    )
    return candidates


def map_tab(stop_df: pd.DataFrame, show_map: bool) -> None:
    """지도 분석 탭을 구성합니다."""
    if not show_map:
        empty_message("사이드바에서 지도 표시 여부가 꺼져 있습니다.")
        return
    deck = create_pydeck_map(stop_df)
    if deck is None:
        empty_message("위치정보 데이터가 없어 지도 분석을 표시할 수 없습니다.")
        return
    st.pydeck_chart(deck, use_container_width=True)


def format_signed_number(value) -> str:
    """증감 인원을 부호와 천 단위 쉼표가 있는 문자열로 바꿉니다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if pd.isna(number):
        return "-"
    return f"{number:+,.0f}"


def prepare_yoy_rank_data(
    yoy_df: pd.DataFrame,
    metric: str,
    top_n: int,
    ascending: bool,
    min_current_value: int | None = None,
) -> tuple[pd.DataFrame, bool, bool]:
    """전년 대비 증감률 표에 필요한 컬럼만 보기 좋게 정리합니다."""
    required = {"stop_name", "year", "month", metric, "previous_year_value", "yoy_growth_rate"}
    if yoy_df.empty or not required.issubset(yoy_df.columns):
        return pd.DataFrame(), False, False

    data = yoy_df.copy()
    filtered = data[data["previous_year_value"] >= YOY_MIN_PREVIOUS_VALUE].copy()
    used_minimum_filter = True
    if filtered.empty:
        filtered = data.copy()
        used_minimum_filter = False

    used_current_filter = min_current_value is not None
    if min_current_value is not None:
        filtered = filtered[filtered[metric] >= min_current_value].copy()

    ranked = filtered.sort_values("yoy_growth_rate", ascending=ascending).head(top_n).copy()
    if ranked.empty:
        return pd.DataFrame(), used_minimum_filter, used_current_filter

    ranked["순위"] = range(1, len(ranked) + 1)
    ranked["연월"] = ranked["year"].astype(int).astype(str) + "-" + ranked["month"].astype(int).astype(str).str.zfill(2)
    ranked["증감 인원"] = ranked[metric] - ranked["previous_year_value"]
    display_cols = ["순위", "stop_name", "연월", metric, "previous_year_value", "증감 인원", "yoy_growth_rate"]
    display = ranked[display_cols].rename(
        columns={
            "stop_name": "정류소명",
            metric: "승차 인원",
            "previous_year_value": "전년 동월",
            "yoy_growth_rate": "증감률",
        }
    )
    return display, used_minimum_filter, used_current_filter


def build_target_year_yoy_trend(
    monthly_df: pd.DataFrame,
    metric: str,
    base_year: int = YOY_BASE_YEAR,
    target_year: int = YOY_TARGET_YEAR,
) -> pd.DataFrame:
    """2026년 월별 이용량을 2025년 같은 달과 비교하는 추세 표를 만듭니다."""
    required = {"year", "month", metric}
    if monthly_df.empty or not required.issubset(monthly_df.columns):
        return pd.DataFrame()

    monthly_total = monthly_df.groupby(["year", "month"], as_index=False)[metric].sum()
    base = monthly_total[monthly_total["year"].astype(int) == base_year][["month", metric]].rename(
        columns={metric: "base_value"}
    )
    target = monthly_total[monthly_total["year"].astype(int) == target_year][["month", metric]].rename(
        columns={metric: "target_value"}
    )
    trend = target.merge(base, on="month", how="inner")
    if trend.empty:
        return pd.DataFrame()

    trend = trend.sort_values("month").copy()
    trend["period"] = trend["month"].astype(int).astype(str).str.zfill(2) + "월"
    trend["change"] = trend["target_value"] - trend["base_value"]
    trend["growth_rate"] = trend["change"] / trend["base_value"].replace({0: pd.NA}) * 100
    return trend


def render_target_year_yoy_trend(trend: pd.DataFrame) -> None:
    """2025년 대비 2026년 월별 증감 추세를 차트와 요약 표로 보여줍니다."""
    if trend.empty:
        empty_message(f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 비교 데이터가 없습니다.")
        return

    chart = trend.set_index("period")[["base_value", "target_value"]].rename(
        columns={"base_value": f"{YOY_BASE_YEAR}년", "target_value": f"{YOY_TARGET_YEAR}년"}
    )
    st.subheader(f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 승차 인원")
    st.line_chart(chart)

    rate_chart = trend.set_index("period")["growth_rate"].rename("전년 동월 대비 증감률")
    st.bar_chart(rate_chart)

    display = trend[["period", "base_value", "target_value", "change", "growth_rate"]].rename(
        columns={
            "period": "월",
            "base_value": f"{YOY_BASE_YEAR}년 승차 인원",
            "target_value": f"{YOY_TARGET_YEAR}년 승차 인원",
            "change": "증감 인원",
            "growth_rate": "증감률",
        }
    )
    formatted = display.copy()
    for col in [f"{YOY_BASE_YEAR}년 승차 인원", f"{YOY_TARGET_YEAR}년 승차 인원"]:
        formatted[col] = formatted[col].apply(format_number)
    formatted["증감 인원"] = formatted["증감 인원"].apply(format_signed_number)
    formatted["증감률"] = formatted["증감률"].apply(lambda value: "-" if pd.isna(value) else f"{float(value):+,.1f}%")
    st.dataframe(formatted, use_container_width=True, hide_index=True)


def render_yoy_rank_table(table: pd.DataFrame, positive: bool) -> None:
    """전년 대비 증감률 순위를 발표 화면에 맞는 HTML 표로 보여줍니다."""
    if table.empty:
        empty_message("표시할 전년 대비 증감률 데이터가 없습니다.")
        return

    accent = "#2dd4bf" if positive else "#fb7185"
    badge_text = "증가" if positive else "감소"
    rows = []
    for _, row in table.iterrows():
        rate = row.get("증감률")
        rate_value = None if pd.isna(rate) else float(rate)
        rate_text = "-" if rate_value is None else f"{rate_value:+,.1f}%"
        rate_class = "rate-up" if rate_value is not None and rate_value >= 0 else "rate-down"
        rows.append(
            "<tr>"
            f"<td class='rank'>{int(row['순위'])}</td>"
            f"<td class='name'>{escape(str(row['정류소명']))}</td>"
            f"<td>{escape(str(row['연월']))}</td>"
            f"<td class='number'>{format_number(row['승차 인원'])}</td>"
            f"<td class='number'>{format_number(row['전년 동월'])}</td>"
            f"<td class='number'>{format_signed_number(row['증감 인원'])}</td>"
            f"<td><span class='rate-badge {rate_class}'>{rate_text}</span></td>"
            "</tr>"
        )

    html = f"""
    <style>
    .yoy-table-wrap {{
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 10px;
        overflow: hidden;
        background: rgba(15, 23, 42, 0.28);
    }}
    .yoy-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }}
    .yoy-table thead th {{
        padding: 0.72rem 0.75rem;
        color: #e5e7eb;
        background: linear-gradient(90deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.88));
        border-bottom: 2px solid {accent};
        text-align: left;
        white-space: nowrap;
    }}
    .yoy-table tbody td {{
        padding: 0.68rem 0.75rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.14);
        color: #dbe4ee;
        vertical-align: middle;
    }}
    .yoy-table tbody tr:nth-child(even) {{
        background: rgba(148, 163, 184, 0.06);
    }}
    .yoy-table tbody tr:hover {{
        background: rgba(45, 212, 191, 0.08);
    }}
    .yoy-table .rank {{
        width: 3.3rem;
        color: #94a3b8;
        font-weight: 700;
        text-align: center;
    }}
    .yoy-table .name {{
        color: #f8fafc;
        font-weight: 700;
        max-width: 14rem;
    }}
    .yoy-table .number {{
        text-align: right;
        font-variant-numeric: tabular-nums;
        white-space: nowrap;
    }}
    .rate-badge {{
        display: inline-flex;
        justify-content: center;
        min-width: 5.2rem;
        padding: 0.22rem 0.55rem;
        border-radius: 999px;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
    }}
    .rate-up {{
        color: #14b8a6;
        background: rgba(20, 184, 166, 0.14);
    }}
    .rate-down {{
        color: #fb7185;
        background: rgba(251, 113, 133, 0.14);
    }}
    </style>
    <div class="yoy-table-wrap">
      <table class="yoy-table">
        <thead>
          <tr>
            <th>순위</th>
            <th>정류소명</th>
            <th>연월</th>
            <th>승차 인원</th>
            <th>전년 동월</th>
            <th>증감 인원</th>
            <th>{badge_text}율</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def cluster_hour_columns(patterns: pd.DataFrame) -> list:
    """K-means 패턴 표에서 시간대 컬럼만 골라 정렬합니다."""
    if patterns.empty:
        return []
    return sorted([col for col in patterns.columns if isinstance(col, (int, float))], key=int)


def plot_cluster_pattern_heatmap(patterns: pd.DataFrame):
    """K-means 군집별 시간대 이용 비율을 퍼센트 히트맵으로 보여줍니다."""
    try:
        import plotly.express as px
    except ImportError:
        return None

    hour_cols = cluster_hour_columns(patterns)
    if patterns.empty or not hour_cols:
        return None

    heatmap = patterns.copy()
    heatmap["군집 유형"] = heatmap.apply(
        lambda row: f"{row.get('cluster_name', '군집')} ({int(row.get('cluster', 0))})",
        axis=1,
    )
    heatmap_data = heatmap.set_index("군집 유형")[hour_cols] * 100
    heatmap_data.columns = [f"{int(hour):02d}시" for hour in hour_cols]

    fig = px.imshow(
        heatmap_data,
        aspect="auto",
        color_continuous_scale="YlGnBu",
        title="군집별 시간대 승차 비율",
        labels=dict(x="시간대", y="군집 유형", color="승차 비율(%)"),
    )
    fig.update_traces(
        texttemplate="%{z:.1f}%",
        hovertemplate="군집: %{y}<br>시간대: %{x}<br>승차 비율: %{z:.1f}%<extra></extra>",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def render_cluster_pattern_summary(patterns: pd.DataFrame) -> None:
    """K-means 군집별 시간대 패턴을 퍼센트 단위로 보기 좋게 표시합니다."""
    hour_cols = cluster_hour_columns(patterns)
    if patterns.empty or not hour_cols:
        empty_message("군집별 시간대 패턴을 표시할 수 없습니다.")
        return

    st.subheader("K-means 군집별 시간대 패턴")
    st.caption(
        "각 시간대 값은 인원 수가 아니라 해당 군집의 전체 승차 인원 중 그 시간대가 차지하는 비율입니다. "
        "예를 들어 0.0043은 0.43%, 즉 약 0.4%를 의미합니다."
    )

    card_cols = st.columns(min(len(patterns), 4))
    for idx, (_, row) in enumerate(patterns.iterrows()):
        hour_values = row[hour_cols].astype(float)
        peak_hour = int(hour_values.idxmax())
        peak_share = float(hour_values.max() * 100)
        morning_share = float(row[[hour for hour in hour_cols if 7 <= int(hour) <= 9]].sum() * 100)
        evening_share = float(row[[hour for hour in hour_cols if 17 <= int(hour) <= 20]].sum() * 100)
        cluster_name = str(row.get("cluster_name", "군집"))
        cluster_no = int(row.get("cluster", idx))

        with card_cols[idx % len(card_cols)]:
            st.metric(f"{cluster_name}", f"{peak_hour:02d}시", f"피크 {peak_share:.1f}%")
            st.caption(f"군집 {cluster_no} · 출근 {morning_share:.1f}% · 퇴근 {evening_share:.1f}%")

    fig = plot_cluster_pattern_heatmap(patterns)
    render_plotly_chart(fig, "군집별 시간대 승차 비율 히트맵을 표시할 수 없습니다.")

    display_rows = []
    for _, row in patterns.iterrows():
        hour_values = row[hour_cols].astype(float)
        peak_hour = int(hour_values.idxmax())
        item = {
            "군집": int(row.get("cluster", 0)),
            "군집 유형": row.get("cluster_name", "군집"),
            "최대 시간대": f"{peak_hour:02d}시",
            "출근 비율": f"{float(row[[hour for hour in hour_cols if 7 <= int(hour) <= 9]].sum() * 100):.1f}%",
            "퇴근 비율": f"{float(row[[hour for hour in hour_cols if 17 <= int(hour) <= 20]].sum() * 100):.1f}%",
        }
        for hour in hour_cols:
            item[f"{int(hour):02d}시"] = f"{float(row[hour]) * 100:.1f}%"
        display_rows.append(item)

    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


def monthly_tab(monthly_df: pd.DataFrame, top_n: int) -> None:
    """장기 월별 추세 분석 탭을 구성합니다."""
    if monthly_df.empty or not {"year", "month"}.issubset(monthly_df.columns):
        empty_message("월별 장기 데이터가 없어 장기 추세 분석을 표시할 수 없습니다.")
        return

    metric = "boardings" if "boardings" in monthly_df.columns else "passengers"
    trend = build_target_year_yoy_trend(monthly_df, metric)
    render_target_year_yoy_trend(trend)

    yoy = monthly_yoy_growth(monthly_df, metric=metric)
    if yoy.empty:
        empty_message("전년 같은 달과 비교할 수 있는 데이터가 부족합니다.")
        return

    target_yoy = yoy[yoy["year"].astype(int) == YOY_TARGET_YEAR].copy()
    if target_yoy.empty:
        empty_message(f"{YOY_TARGET_YEAR}년과 {YOY_BASE_YEAR}년을 비교할 수 있는 정류소별 월별 데이터가 없습니다.")
        return

    available_months = ", ".join(target_yoy["month"].dropna().astype(int).drop_duplicates().sort_values().astype(str) + "월")
    st.caption(
        f"아래 순위는 {YOY_TARGET_YEAR}년 {available_months} 승차 인원을 "
        f"{YOY_BASE_YEAR}년 같은 달과 비교한 결과만 사용합니다."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader(f"{YOY_TARGET_YEAR}년 이용량 증가율 TOP {top_n}")
        increase_table, used_filter, _ = prepare_yoy_rank_data(target_yoy, metric, top_n, ascending=False)
        render_yoy_rank_table(increase_table, positive=True)
    with col2:
        st.subheader(f"{YOY_TARGET_YEAR}년 이용량 감소율 TOP {top_n}")
        decrease_table, _, used_current_filter = prepare_yoy_rank_data(
            target_yoy,
            metric,
            top_n,
            ascending=True,
            min_current_value=YOY_MIN_CURRENT_VALUE_FOR_DECREASE,
        )
        render_yoy_rank_table(decrease_table, positive=False)
    if used_filter:
        st.caption(
            f"전년 동월 승차 인원이 {YOY_MIN_PREVIOUS_VALUE:,}명 미만인 행은 증감률이 과장될 수 있어 순위 표에서 제외했습니다."
        )
    if used_current_filter:
        st.caption(
            f"감소율 표에서는 현재 월 승차 인원이 {YOY_MIN_CURRENT_VALUE_FOR_DECREASE:,}명 미만인 행을 "
            "정류소 운영 중단, 명칭 변경, ID 매칭 또는 자료 누락 확인 대상으로 보고 순위에서 제외했습니다."
        )


def weather_bus_tab(bundle: dict, monthly_df: pd.DataFrame) -> None:
    """날씨와 버스 이용량의 월별 연관 분석 탭을 구성합니다."""
    weather_monthly = bundle.get("weather_monthly", pd.DataFrame())

    st.subheader("분석 기준")
    st.info(
        "월별 날씨 CSV를 표준 컬럼으로 정리한 뒤, 버스 이용량도 먼저 월별 한 행으로 집계해서 같은 연도·월 기준으로 결합합니다. "
        "따라서 상관계수는 정류소별 반복 행이 아니라 월별 관측치 기준으로 계산됩니다."
    )

    if weather_monthly.empty:
        empty_message("월별 날씨 날짜 컬럼을 변환할 수 없어 날씨 지표를 만들 수 없습니다.")
        return
    if monthly_df.empty or not {"year", "month"}.issubset(monthly_df.columns):
        empty_message("월별 버스 데이터가 없어 날씨 월별 자료와 결합하지 않습니다.")
        return

    weather_options = available_weather_variables(weather_monthly)
    if not weather_options:
        empty_message("분석 가능한 날씨 변수가 없습니다.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        weather_col = st.selectbox(
            "날씨 변수 선택",
            list(weather_options.keys()),
            format_func=lambda col: weather_options[col],
            help="선택한 변수는 아래 상관관계 카드, 이중축 추세 그래프, 선택 변수 산점도에 반영됩니다.",
        )
    with col2:
        stop_options = ["전체 대구"]
        if "stop_name" in monthly_df.columns:
            stop_options += sorted(monthly_df["stop_name"].dropna().astype(str).unique())
        selected_stop = st.selectbox("전체 대구 또는 정류소 선택", stop_options)
    with col3:
        st.metric("날씨 관측 월 수", len(weather_monthly))

    bus_monthly = aggregate_bus_monthly(monthly_df, selected_stop)
    merged = merge_bus_weather_monthly(bus_monthly, weather_monthly)
    if merged.empty:
        empty_message("버스 월별 데이터와 날씨 월별 데이터의 기간이 겹치지 않아 결합하지 않았습니다.")
        return

    periods = merged["period"].dropna().tolist()
    if len(periods) >= 2:
        start_period, end_period = st.select_slider("분석 기간 선택", options=periods, value=(periods[0], periods[-1]))
        filtered = merged[(merged["period"] >= start_period) & (merged["period"] <= end_period)].copy()
    else:
        filtered = merged.copy()
        st.caption("겹치는 관측 월이 1개뿐이라 기간 슬라이더를 표시하지 않습니다.")

    if filtered.empty:
        empty_message("선택한 기간에 결합된 데이터가 없습니다.")
        return

    if len(filtered) < 12:
        st.warning("관측 월 수가 12개월 미만이어서 상관관계 해석의 신뢰성이 낮습니다.")

    correlation_df = calculate_weather_correlations(filtered)
    selected_corr = correlation_df[correlation_df["weather_column"] == weather_col]
    selected_corr = selected_corr.iloc[0] if not selected_corr.empty else pd.Series(dtype=object)

    st.subheader(f"{weather_options[weather_col]}와 버스 이용량")
    st.caption(
        "Pearson은 두 값의 선형적인 움직임을, Spearman은 값의 순위가 함께 움직이는 정도를 봅니다. "
        "둘 다 인과관계가 아니라 월별 경향을 보는 지표입니다."
    )
    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Pearson", format_number(selected_corr.get("Pearson 상관계수"), 2))
    with metric_cols[1]:
        st.metric("Spearman", format_number(selected_corr.get("Spearman 상관계수"), 2))
    with metric_cols[2]:
        show_metric("관측 월 수", selected_corr.get("관측 월 수"))
    st.write(selected_weather_correlation_summary(correlation_df, weather_col))
    st.caption("상관관계는 두 변수가 함께 움직이는 정도를 보여줄 뿐이며 인과관계를 의미하지 않습니다.")

    st.download_button(
        "날씨 연관 분석 결과 CSV 다운로드",
        data=make_download_csv(filtered),
        file_name="bus_weather_monthly_merged.csv",
        mime="text/csv",
    )

    col4, col5 = st.columns(2)
    with col4:
        fig = plot_weather_monthly_line(filtered, "boardings", "월별 승차 인원", "월별 버스 승차 인원 추세")
        render_plotly_chart(fig, "월별 버스 승차 인원 추세를 표시할 수 없습니다.")
    with col5:
        fig = plot_weather_monthly_line(filtered, "monthly_avg_temp", "월평균 기온", "월별 기온 추세")
        render_plotly_chart(fig, "월평균 기온 컬럼이 없어 기온 추세를 표시할 수 없습니다.")

    fig = plot_weather_bus_dual_line(filtered, weather_col, weather_options[weather_col])
    render_plotly_chart(fig, "버스 이용량과 날씨 추세 비교 그래프를 표시할 수 없습니다.")

    selected_scatter = plot_weather_scatter(
        filtered,
        weather_col,
        weather_options[weather_col],
        f"{weather_options[weather_col]}와 월별 승차 인원의 산점도",
    )
    render_plotly_chart(selected_scatter, "선택한 날씨 변수 산점도를 표시할 수 없습니다.")

    col6, col7 = st.columns(2)
    with col6:
        fig = plot_weather_scatter(
            filtered,
            "monthly_avg_temp",
            "월평균 기온",
            "기온과 월별 승차 인원의 산점도",
        )
        render_plotly_chart(fig, "기온 산점도를 표시할 수 없습니다.")
    with col7:
        fig = plot_weather_scatter(
            filtered,
            "monthly_precipitation",
            "월 누적 강수량",
            "강수량과 월별 승차 인원의 산점도",
        )
        render_plotly_chart(fig, "강수량 산점도를 표시할 수 없습니다.")

    col8, col9 = st.columns(2)
    with col8:
        fig = plot_season_bus_box(filtered)
        render_plotly_chart(fig, "계절별 이용량 박스플롯을 표시할 수 없습니다.")
    with col9:
        fig = plot_weather_correlation_heatmap(correlation_df)
        render_plotly_chart(fig, "날씨 변수 상관관계 히트맵을 표시할 수 없습니다.")

    st.subheader("날씨 변수별 상관계수")
    display_corr = correlation_df.drop(columns=["weather_column"], errors="ignore").copy()
    for col in ["Pearson 상관계수", "Spearman 상관계수"]:
        if col in display_corr.columns:
            display_corr[col] = display_corr[col].round(2)
    st.dataframe(display_corr, use_container_width=True, hide_index=True)

    st.subheader("날씨 연관 분석 한계")
    st.markdown(
        """
- 공휴일과 방학은 월별 이용량에 영향을 줄 수 있지만 현재 분석에는 별도로 반영하지 않았습니다.
- 노선 개편과 배차 변화가 있으면 날씨와 무관하게 이용량이 달라질 수 있습니다.
- 지역 행사와 대형 집객 시설의 일정은 월별 수요 변화를 만들 수 있습니다.
- 유가 변화와 대체 교통수단 이용 변화는 버스 이용량에 영향을 줄 수 있습니다.
- 코로나19와 같은 외부 요인은 장기 추세를 크게 바꿀 수 있습니다.
- 월별 자료에서는 개별 강수일의 즉각적인 영향을 확인하기 어렵습니다.
"""
    )


def data_limit_tab(bundle: dict) -> None:
    """데이터 점검 결과와 분석 한계를 표시합니다."""
    st.subheader("분석 한계")
    st.markdown(
        """
- 하차 태그를 하지 않은 승객이 있을 수 있어 실제 하차 인원과 차이가 날 수 있습니다.
- 현금 승차가 데이터에서 제외될 수 있습니다.
- 경유 노선 수가 많다고 실제 배차 횟수가 많은 것은 아닙니다.
- 버스 배차 간격과 차량 크기 데이터가 없습니다.
- 이용객 수만으로 실제 차량 내부 혼잡도를 확정할 수 없습니다.
- 특정 정류소의 이용량이 높은 이유는 주변 학교, 상권, 병원, 환승센터 등 추가 정보가 필요합니다.
- 분석 결과는 노선 부족을 확정하는 것이 아니라 추가 검토 후보를 찾는 것입니다.
- 데이터 수집 기간에 따라 계절성과 일시적 이벤트의 영향을 받을 수 있습니다.
- 공휴일과 방학, 노선 개편, 지역 행사, 유가 변화, 코로나19 등의 외부 요인이 이용량에 영향을 줄 수 있습니다.
- 월별 자료에서는 개별 강수일의 즉각적인 영향을 확인하기 어렵습니다.
"""
    )


def main() -> None:
    """Streamlit 앱의 시작점입니다."""
    st.set_page_config(page_title="대구 시내버스 정류소 분석", layout="wide")
    ensure_directories(BASE_DIR)
    set_korean_font()

    st.title("대구 시내버스 정류소 이용 수요 및 노선 공급 불균형 분석")
    st.write(
        "대구 시내버스 정류소의 시간대별 승하차 데이터를 분석하여 수요가 집중되는 시간과 지역을 확인하고, "
        "이용 수요 대비 경유 노선 수가 상대적으로 적은 정류소 후보를 탐색합니다."
    )

    bundle = load_project_data(str(DATA_DIR), data_fingerprint(DATA_DIR))
    stop_df = bundle.get("stop_summary", pd.DataFrame())
    hourly_df = bundle.get("hourly_summary", pd.DataFrame())
    monthly_df = bundle.get("monthly_summary", pd.DataFrame())

    if bundle.get("processed_cache_missing"):
        st.error("정제된 분석 CSV가 없습니다. PowerShell에서 `python prepare_data.py`를 먼저 실행한 뒤 앱을 새로고침하세요.")
        st.code("python prepare_data.py", language="powershell")
        return

    if bundle.get("processed_cache_stale"):
        st.warning("data 폴더의 원본 CSV가 정제 결과보다 최신입니다. 최신 데이터 반영이 필요하면 `python prepare_data.py`를 다시 실행하세요.")

    if stop_df.empty:
        st.warning("정제된 분석 CSV에 표시할 정류소 데이터가 없습니다. 원본 CSV를 확인한 뒤 `python prepare_data.py`를 다시 실행하세요.")
        data_limit_tab(bundle)
        return

    filters = sidebar_filters(stop_df, monthly_df)
    filtered_stop = filter_stop_data(stop_df, filters)
    filtered_hourly = filter_hourly_data(hourly_df, filtered_stop, filters["hours"])

    if filters["show_static"]:
        if st.sidebar.button("정적 그래프 PNG 저장"):
            saved_files, skipped = save_static_figures(filtered_stop, filtered_hourly, monthly_df, FIGURE_DIR)
            st.sidebar.success(f"{len(saved_files)}개 그래프를 outputs/figures에 저장했습니다.")
            for message in skipped:
                st.sidebar.caption(message)

    if st.sidebar.button("현재 분석 결과 CSV 저장"):
        save_outputs_on_request(bundle, filtered_stop, filtered_hourly)
        st.sidebar.success("outputs/processed에 저장했습니다.")

    analysis_tables = create_analysis_tables(filtered_stop, filtered_hourly, monthly_df, top_n=filters["top_n"])

    with st.expander("핵심 분석 결과 표 보기"):
        for name, table in analysis_tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                st.write(name)
                st.dataframe(table, use_container_width=True)

    tabs = st.tabs(
        [
            "전체 현황",
            "시간대별 분석",
            "정류소별 분석",
            "수요·공급 불균형 분석",
            "지도 분석",
            "장기 추세 분석",
            "날씨와 버스 이용",
            "데이터 및 분석 한계",
        ]
    )

    with tabs[0]:
        overview_tab(filtered_stop, filtered_hourly, filters["top_n"], filters["metric_col"])
    with tabs[1]:
        hourly_tab(filtered_stop, filtered_hourly, filters["top_n"], filters["metric_col"])
    with tabs[2]:
        stop_detail_tab(filtered_stop, hourly_df, monthly_df)
    with tabs[3]:
        candidates = imbalance_tab(filtered_stop, filters["top_n"])
    with tabs[4]:
        map_tab(filtered_stop, filters["show_map"])
    with tabs[5]:
        monthly_tab(monthly_df, filters["top_n"])
        cluster_result = kmeans_cluster_hourly_patterns(filtered_hourly)
        if cluster_result.get("message"):
            st.caption(cluster_result["message"])
        if isinstance(cluster_result.get("patterns"), pd.DataFrame) and not cluster_result["patterns"].empty:
            render_cluster_pattern_summary(cluster_result["patterns"])
    with tabs[6]:
        weather_bus_tab(bundle, monthly_df)
    with tabs[7]:
        data_limit_tab(bundle)


if __name__ == "__main__":
    main()
