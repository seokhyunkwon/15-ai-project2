from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.analysis import monthly_yoy_growth
from src.utils import set_korean_font


LABELS = {
    "boardings": "승차 인원",
    "alightings": "하차 인원",
    "total_riders": "전체 이용객",
    "boardings_per_route": "노선당 승차 인원",
    "morning_boardings": "출근 시간 승차 인원",
    "evening_boardings": "퇴근 시간 승차 인원",
}


def metric_label(metric: str) -> str:
    """표준 컬럼명을 한글 축 이름으로 바꿉니다."""
    return LABELS.get(metric, metric)


def add_unique_stop_label(data: pd.DataFrame) -> pd.DataFrame:
    """같은 정류소명이 여러 행에 있을 때 그래프 라벨이 겹치지 않도록 보조 라벨을 만듭니다."""
    labeled = data.copy()
    labeled["stop_label"] = labeled["stop_name"].astype(str)
    duplicate_names = labeled["stop_label"].duplicated(keep=False)
    if duplicate_names.any():
        if "stop_id" in labeled.columns:
            suffix = labeled["stop_id"].fillna("").astype(str).str.replace(r"\.0+$", "", regex=True)
            labeled.loc[duplicate_names, "stop_label"] = (
                labeled.loc[duplicate_names, "stop_label"] + " (" + suffix.loc[duplicate_names] + ")"
            )
        elif "district" in labeled.columns:
            suffix = labeled["district"].fillna("구군 미상").astype(str)
            labeled.loc[duplicate_names, "stop_label"] = (
                labeled.loc[duplicate_names, "stop_label"] + " (" + suffix.loc[duplicate_names] + ")"
            )
    return labeled


def get_plotly_express():
    """Plotly가 설치된 경우에만 plotly.express를 불러옵니다."""
    try:
        import plotly.express as px
    except ImportError:
        return None
    return px


def get_plotly_graph_objects():
    """Plotly가 설치된 경우에만 graph_objects를 불러옵니다."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        return None
    return go


def plot_top_stops(stop_df: pd.DataFrame, metric: str = "boardings", n: int = 10, title: str = ""):
    """정류소별 상위 N개 막대그래프를 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if stop_df.empty or metric not in stop_df.columns or "stop_name" not in stop_df.columns:
        return None
    data = stop_df.dropna(subset=[metric]).sort_values(metric, ascending=False).head(n).copy()
    if data.empty:
        return None
    data = add_unique_stop_label(data)
    fig = px.bar(
        data.sort_values(metric),
        x=metric,
        y="stop_label",
        orientation="h",
        color="district" if "district" in data.columns else None,
        title=title or f"정류소별 {metric_label(metric)} TOP {n}",
        labels={metric: metric_label(metric), "stop_label": "정류소명", "district": "구·군"},
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_district_bar(stop_df: pd.DataFrame, metric: str = "boardings", title: str = ""):
    """구·군별 이용량 막대그래프를 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if stop_df.empty or "district" not in stop_df.columns or metric not in stop_df.columns:
        return None
    data = stop_df.groupby("district", as_index=False)[metric].sum().sort_values(metric, ascending=False)
    if data.empty:
        return None
    fig = px.bar(
        data,
        x="district",
        y=metric,
        title=title or f"구·군별 {metric_label(metric)}",
        labels={"district": "구·군", metric: metric_label(metric)},
        text=metric,
    )
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_hourly_line(hourly_df: pd.DataFrame, metric: str = "boardings", title: str = ""):
    """시간대별 전체 이용량 선그래프를 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if hourly_df.empty or "hour" not in hourly_df.columns or metric not in hourly_df.columns:
        return None
    data = hourly_df.groupby("hour", as_index=False)[metric].sum().sort_values("hour")
    if data.empty:
        return None
    fig = px.line(
        data,
        x="hour",
        y=metric,
        markers=True,
        title=title or f"시간대별 {metric_label(metric)}",
        labels={"hour": "시간대", metric: metric_label(metric)},
    )
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_board_alight_line(hourly_df: pd.DataFrame):
    """시간대별 승차·하차 비교 선그래프를 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if hourly_df.empty or "hour" not in hourly_df.columns:
        return None
    required = [col for col in ["boardings", "alightings"] if col in hourly_df.columns]
    if len(required) < 2:
        return None
    data = hourly_df.groupby("hour", as_index=False)[required].sum().sort_values("hour")
    melted = data.melt(id_vars="hour", value_vars=required, var_name="구분", value_name="인원")
    melted["구분"] = melted["구분"].map({"boardings": "승차", "alightings": "하차"})
    fig = px.line(melted, x="hour", y="인원", color="구분", markers=True, title="시간대별 승차·하차 비교")
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_stop_hour_heatmap(hourly_df: pd.DataFrame, stop_df: pd.DataFrame, metric: str = "boardings", top_n: int = 20):
    """정류소 × 시간대 히트맵을 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if hourly_df.empty or metric not in hourly_df.columns or "hour" not in hourly_df.columns or "stop_name" not in hourly_df.columns:
        return None
    if not stop_df.empty and metric in stop_df.columns and "_merge_key" in stop_df.columns:
        top_keys = stop_df.sort_values(metric, ascending=False)["_merge_key"].head(top_n)
        data = hourly_df[hourly_df["_merge_key"].isin(top_keys)].copy()
    else:
        data = hourly_df.copy()
    if data.empty:
        return None
    pivot = data.pivot_table(index="stop_name", columns="hour", values=metric, aggfunc="sum").fillna(0)
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"정류소 × 시간대 {metric_label(metric)} 히트맵",
        labels=dict(x="시간대", y="정류소", color=metric_label(metric)),
    )
    fig.update_layout(height=560, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_district_hour_heatmap(hourly_df: pd.DataFrame, metric: str = "boardings"):
    """구·군 × 시간대 히트맵을 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if hourly_df.empty or metric not in hourly_df.columns or "hour" not in hourly_df.columns or "district" not in hourly_df.columns:
        return None
    pivot = hourly_df.pivot_table(index="district", columns="hour", values=metric, aggfunc="sum").fillna(0)
    if pivot.empty:
        return None
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=f"구·군 × 시간대 {metric_label(metric)} 히트맵",
        labels=dict(x="시간대", y="구·군", color=metric_label(metric)),
    )
    fig.update_layout(height=500, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_stop_comparison_line(
    hourly_df: pd.DataFrame,
    stop_names: list[str],
    metric: str = "boardings",
    include_average: bool = False,
):
    """선택한 정류소들의 시간대 패턴을 비교합니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if hourly_df.empty or not stop_names or metric not in hourly_df.columns or "stop_name" not in hourly_df.columns:
        return None
    selected = hourly_df[hourly_df["stop_name"].astype(str).isin(stop_names)].copy()
    if selected.empty:
        return None
    data = selected.groupby(["stop_name", "hour"], as_index=False)[metric].sum()
    if include_average:
        average = hourly_df.groupby("hour", as_index=False)[metric].mean()
        average["stop_name"] = "전체 정류소 평균"
        data = pd.concat([data, average], ignore_index=True)
    fig = px.line(
        data,
        x="hour",
        y=metric,
        color="stop_name",
        markers=True,
        title="선택 정류소 시간대 패턴 비교",
        labels={"hour": "시간대", metric: metric_label(metric), "stop_name": "정류소"},
    )
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_type_pie(stop_df: pd.DataFrame):
    """정류소 유형별 비율 파이 차트를 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if stop_df.empty or "stop_type" not in stop_df.columns:
        return None
    data = stop_df["stop_type"].value_counts().reset_index()
    data.columns = ["stop_type", "count"]
    if data.empty:
        return None
    fig = px.pie(data, names="stop_type", values="count", title="정류소 유형별 비율")
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_supply_demand_scatter(stop_df: pd.DataFrame, thresholds: dict):
    """경유 노선 수와 전체 승차 인원 산점도를 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    required = {"route_count", "boardings", "boardings_per_route"}
    if stop_df.empty or not required.issubset(stop_df.columns):
        return None
    data = stop_df.dropna(subset=list(required)).copy()
    data = data[data["route_count"] > 0]
    if data.empty:
        return None
    fig = px.scatter(
        data,
        x="route_count",
        y="boardings",
        size="boardings_per_route",
        color="stop_type" if "stop_type" in data.columns else None,
        hover_data={
            "stop_name": True,
            "district": True if "district" in data.columns else False,
            "boardings": ":,",
            "route_count": ":,",
            "boardings_per_route": ":,.1f",
            "peak_hour_label": True if "peak_hour_label" in data.columns else False,
            "stop_type": True if "stop_type" in data.columns else False,
        },
        title="경유 노선 수와 전체 승차 인원",
        labels={
            "route_count": "경유 노선 수",
            "boardings": "전체 승차 인원",
            "boardings_per_route": "노선당 승차 인원",
            "stop_type": "정류소 유형",
        },
    )
    route_line = thresholds.get("route_threshold")
    demand_line = thresholds.get("demand_threshold")
    if pd.notna(route_line):
        fig.add_vline(x=route_line, line_dash="dash", line_color="gray")
    if pd.notna(demand_line):
        fig.add_hline(y=demand_line, line_dash="dash", line_color="gray")
    if pd.notna(route_line) and pd.notna(demand_line):
        fig.add_shape(
            type="rect",
            x0=0,
            x1=route_line,
            y0=demand_line,
            y1=data["boardings"].max(),
            fillcolor="rgba(255, 99, 71, 0.12)",
            line_width=0,
            layer="below",
        )
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def create_pydeck_map(stop_df: pd.DataFrame):
    """위도·경도가 있는 경우 pydeck 지도 객체를 만듭니다."""
    try:
        import pydeck as pdk
    except ImportError:
        return None
    required = {"lat", "lon", "boardings"}
    if stop_df.empty or not required.issubset(stop_df.columns):
        return None
    data = stop_df.dropna(subset=["lat", "lon"]).copy()
    if data.empty:
        return None
    per_route = data.get("boardings_per_route", pd.Series(0, index=data.index)).fillna(0)
    boardings = data["boardings"].fillna(0).clip(lower=0)
    boardings_base = boardings.quantile(0.95)
    if pd.isna(boardings_base) or boardings_base <= 0:
        boardings_base = boardings.max() if boardings.max() > 0 else 1
    route_base = per_route.quantile(0.95)
    if pd.isna(route_base) or route_base <= 0:
        route_base = per_route.max() if per_route.max() > 0 else 1

    # 지나치게 큰 원이 지도를 덮지 않도록 95분위 기준으로 반경을 제한합니다.
    data["radius"] = (40 + np.sqrt((boardings / boardings_base).clip(0, 1)) * 180).round(1)
    data["color_value"] = ((per_route / route_base).clip(0, 1) * 170 + 55).astype(int)
    data["fill_color"] = data["color_value"].apply(lambda value: [int(value), 96, 120, 125])

    center_lat = float(data["lat"].mean())
    center_lon = float(data["lon"].mean())
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=data,
        get_position="[lon, lat]",
        get_radius="radius",
        get_fill_color="fill_color",
        pickable=True,
        auto_highlight=True,
        stroked=True,
        get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=1,
    )
    tooltip = {
        "html": """
        <b>{stop_name}</b><br/>
        구·군: {district}<br/>
        승차 인원: {boardings}<br/>
        하차 인원: {alightings}<br/>
        경유 노선 수: {route_count}<br/>
        노선당 승차 인원: {boardings_per_route}<br/>
        정류소 유형: {stop_type}
        """,
        "style": {"backgroundColor": "white", "color": "black"},
    }
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0)
    return pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)


def plot_weather_bus_dual_line(data: pd.DataFrame, weather_col: str, weather_label: str):
    """월별 버스 승차 인원과 선택한 날씨 지표를 이중축 선그래프로 비교합니다."""
    go = get_plotly_graph_objects()
    if go is None or data.empty or weather_col not in data.columns or "boardings" not in data.columns:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["period"],
            y=data["boardings"],
            mode="lines+markers",
            name="월별 버스 승차 인원",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["period"],
            y=data[weather_col],
            mode="lines+markers",
            name=weather_label,
            yaxis="y2",
        )
    )
    fig.update_layout(
        title="월별 버스 이용량과 날씨 추세 비교",
        xaxis_title="연도·월",
        yaxis=dict(title="승차 인원"),
        yaxis2=dict(title=weather_label, overlaying="y", side="right"),
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_weather_monthly_line(data: pd.DataFrame, y_col: str, y_label: str, title: str):
    """월별 날씨 또는 버스 지표의 추세 선그래프를 만듭니다."""
    px = get_plotly_express()
    if px is None or data.empty or y_col not in data.columns:
        return None
    fig = px.line(data, x="period", y=y_col, markers=True, title=title, labels={"period": "연도·월", y_col: y_label})
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_weather_scatter(data: pd.DataFrame, weather_col: str, weather_label: str, title: str):
    """날씨 변수와 월별 버스 승차 인원의 산점도를 만듭니다."""
    px = get_plotly_express()
    if px is None or data.empty or weather_col not in data.columns or "boardings" not in data.columns:
        return None
    fig = px.scatter(
        data,
        x=weather_col,
        y="boardings",
        color="season" if "season" in data.columns else None,
        hover_data=["period"],
        title=title,
        labels={weather_col: weather_label, "boardings": "월별 승차 인원", "season": "계절", "period": "연도·월"},
    )
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_season_bus_box(data: pd.DataFrame):
    """계절별 월별 버스 이용량 분포를 박스플롯으로 보여줍니다."""
    px = get_plotly_express()
    if px is None or data.empty or "season" not in data.columns or "boardings" not in data.columns:
        return None
    order = ["봄", "여름", "가을", "겨울"]
    fig = px.box(
        data,
        x="season",
        y="boardings",
        category_orders={"season": order},
        points="all",
        title="계절별 버스 이용량 비교",
        labels={"season": "계절", "boardings": "월별 승차 인원"},
    )
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def plot_weather_correlation_heatmap(correlation_df: pd.DataFrame):
    """날씨 변수별 Pearson, Spearman 상관계수를 히트맵으로 표시합니다."""
    px = get_plotly_express()
    if px is None or correlation_df.empty:
        return None
    value_cols = ["Pearson 상관계수", "Spearman 상관계수"]
    available = [col for col in value_cols if col in correlation_df.columns]
    if not available:
        return None
    heatmap_data = correlation_df.set_index("날씨 변수")[available]
    fig = px.imshow(
        heatmap_data,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="날씨 변수 상관관계 히트맵",
        labels=dict(x="상관계수 종류", y="날씨 변수", color="상관계수"),
    )
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def save_current_figure(path: Path) -> None:
    """현재 Matplotlib 그림을 고해상도 PNG로 저장하고 닫습니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_static_figures(
    stop_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 10,
) -> tuple[list[str], list[str]]:
    """발표 PPT에 사용할 정적 그래프 PNG 파일을 저장합니다."""
    set_korean_font()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    skipped: list[str] = []

    if not stop_df.empty and {"stop_name", "boardings"}.issubset(stop_df.columns):
        data = stop_df.sort_values("boardings", ascending=False).head(top_n)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=data, x="boardings", y="stop_name")
        plt.title("정류소별 승차 인원 TOP 10")
        plt.xlabel("승차 인원")
        plt.ylabel("정류소")
        path = output_dir / "01_stop_boardings_top10.png"
        save_current_figure(path)
        saved.append(str(path))
    else:
        skipped.append("정류소별 승차 인원 TOP 10: 필요한 컬럼이 없습니다.")

    if not stop_df.empty and {"district", "boardings"}.issubset(stop_df.columns):
        data = stop_df.groupby("district", as_index=False)["boardings"].sum().sort_values("boardings", ascending=False)
        plt.figure(figsize=(9, 5))
        sns.barplot(data=data, x="district", y="boardings")
        plt.title("구·군별 승차 인원")
        plt.xlabel("구·군")
        plt.ylabel("승차 인원")
        plt.xticks(rotation=30)
        path = output_dir / "02_district_boardings.png"
        save_current_figure(path)
        saved.append(str(path))
    else:
        skipped.append("구·군별 승차 인원: 필요한 컬럼이 없습니다.")

    if not hourly_df.empty and {"hour", "boardings"}.issubset(hourly_df.columns):
        data = hourly_df.groupby("hour", as_index=False)["boardings"].sum().sort_values("hour")
        plt.figure(figsize=(9, 5))
        sns.lineplot(data=data, x="hour", y="boardings", marker="o")
        plt.title("시간대별 전체 승차 인원")
        plt.xlabel("시간대")
        plt.ylabel("승차 인원")
        path = output_dir / "03_hourly_boardings.png"
        save_current_figure(path)
        saved.append(str(path))

        heat = hourly_df.pivot_table(index="stop_name", columns="hour", values="boardings", aggfunc="sum").fillna(0)
        heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).head(20).index]
        plt.figure(figsize=(12, 8))
        sns.heatmap(heat, cmap="YlOrRd")
        plt.title("정류소 × 시간대 히트맵")
        plt.xlabel("시간대")
        plt.ylabel("정류소")
        path = output_dir / "04_stop_hour_heatmap.png"
        save_current_figure(path)
        saved.append(str(path))
    else:
        skipped.append("시간대 그래프 및 히트맵: 시간대별 승차 데이터가 없습니다.")

    if not stop_df.empty and {"route_count", "boardings"}.issubset(stop_df.columns):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=stop_df, x="route_count", y="boardings", hue="stop_type" if "stop_type" in stop_df.columns else None)
        plt.title("경유 노선 수와 승차 인원 산점도")
        plt.xlabel("경유 노선 수")
        plt.ylabel("승차 인원")
        path = output_dir / "05_route_count_boardings_scatter.png"
        save_current_figure(path)
        saved.append(str(path))
    else:
        skipped.append("경유 노선 수와 승차 인원 산점도: 필요한 컬럼이 없습니다.")

    if not stop_df.empty and {"stop_name", "boardings_per_route"}.issubset(stop_df.columns):
        data = stop_df.sort_values("boardings_per_route", ascending=False).head(top_n)
        plt.figure(figsize=(9, 6))
        sns.barplot(data=data, x="boardings_per_route", y="stop_name")
        plt.title("노선당 승차 인원 TOP 10")
        plt.xlabel("노선당 승차 인원")
        plt.ylabel("정류소")
        path = output_dir / "06_boardings_per_route_top10.png"
        save_current_figure(path)
        saved.append(str(path))
    else:
        skipped.append("노선당 승차 인원 TOP 10: 필요한 컬럼이 없습니다.")

    if not stop_df.empty and "stop_type" in stop_df.columns:
        data = stop_df["stop_type"].value_counts().reset_index()
        data.columns = ["stop_type", "count"]
        plt.figure(figsize=(8, 5))
        sns.barplot(data=data, x="stop_type", y="count")
        plt.title("정류소 유형별 개수")
        plt.xlabel("정류소 유형")
        plt.ylabel("정류소 수")
        path = output_dir / "07_stop_type_count.png"
        save_current_figure(path)
        saved.append(str(path))

        compare = stop_df[stop_df["stop_type"].isin(["출근형", "퇴근형"])]
        if not compare.empty and "boardings" in compare.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=compare, x="stop_type", y="boardings")
            plt.title("출근형·퇴근형 정류소 비교")
            plt.xlabel("정류소 유형")
            plt.ylabel("승차 인원")
            path = output_dir / "08_commute_type_compare.png"
            save_current_figure(path)
            saved.append(str(path))
    else:
        skipped.append("정류소 유형별 그래프: 정류소 유형이 없습니다.")

    if not monthly_df.empty and {"year", "month"}.issubset(monthly_df.columns):
        metric = "boardings" if "boardings" in monthly_df.columns else "passengers"
        if metric in monthly_df.columns:
            data = monthly_df.groupby(["year", "month"], as_index=False)[metric].sum()
            data["period"] = data["year"].astype(int).astype(str) + "-" + data["month"].astype(int).astype(str).str.zfill(2)
            plt.figure(figsize=(10, 5))
            sns.lineplot(data=data, x="period", y=metric, marker="o")
            plt.title("월별 장기 추세")
            plt.xlabel("연도·월")
            plt.ylabel("이용량")
            plt.xticks(rotation=45)
            path = output_dir / "09_monthly_trend.png"
            save_current_figure(path)
            saved.append(str(path))

            yoy = monthly_yoy_growth(monthly_df, metric)
            if not yoy.empty:
                top_up = yoy.sort_values("yoy_growth_rate", ascending=False).head(top_n)
                top_down = yoy.sort_values("yoy_growth_rate").head(top_n)
                combined = pd.concat([top_up.assign(type="증가율 TOP"), top_down.assign(type="감소율 TOP")])
                plt.figure(figsize=(10, 6))
                sns.barplot(data=combined, x="yoy_growth_rate", y="stop_name", hue="type")
                plt.title("이용량 증가율 및 감소율 TOP 10")
                plt.xlabel("전년 대비 증감률(%)")
                plt.ylabel("정류소")
                path = output_dir / "10_yoy_growth_top10.png"
                save_current_figure(path)
                saved.append(str(path))
        else:
            skipped.append("월별 장기 추세: 이용량 컬럼이 없습니다.")
    else:
        skipped.append("월별 장기 추세: 월별 데이터가 없습니다.")

    return saved, skipped
