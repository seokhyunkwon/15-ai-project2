from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from src.analysis import monthly_yoy_growth
from src.utils import set_korean_font


DARK_BG = "#C0C0C0"
DARK_SURFACE = "#FFFFFF"
DARK_TEXT = "#000000"
DARK_MUTED = "#808080"
DARK_STROKE = "#000000"
ACCENT_BLUE = "#0000FF"
ACCENT_BLUE_DEEP = "#000080"
SUCCESS_GREEN = "#00AA00"
ALERT_ROSE = "#FF0000"
RETRO_YELLOW = "#FFFF00"
RETRO_PANEL = "#FFFFCC"
PLOTLY_COLORWAY = [ACCENT_BLUE, SUCCESS_GREEN, ALERT_ROSE, RETRO_YELLOW, "#800080", "#008080", "#808080"]

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


def readable_bar_text_color(marker_color) -> str:
    """노란 막대 위 숫자는 검정, 그 외 강한 색 막대 위 숫자는 흰색으로 표시합니다."""
    if isinstance(marker_color, (list, tuple, np.ndarray, pd.Series)) and len(marker_color) > 0:
        marker_color = marker_color[0]
    color = str(marker_color or "").lower()
    if color in {"#ffff00", "yellow", "rgb(255, 255, 0)", "rgba(255, 255, 0, 1)"}:
        return "#000000"
    return "#FFFFFF"


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


def apply_dark_plotly_theme(fig):
    """모든 Plotly 그래프에 동일한 90년대 레트로 테마를 적용합니다."""
    if fig is None:
        return None
    for trace in fig.data:
        trace_type = getattr(trace, "type", "")
        if trace_type == "scatter":
            trace.update(line=dict(width=3), marker=dict(size=7, line=dict(width=2, color=DARK_STROKE)))
        if trace_type == "bar":
            trace.update(marker_line_width=2, marker_line_color=DARK_STROKE, opacity=1)
            trace.update(textfont=dict(color=readable_bar_text_color(getattr(trace.marker, "color", None)), size=12))
    current_height = fig.layout.height
    if current_height is None or current_height < 520:
        fig.update_layout(height=520)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_SURFACE,
        autosize=True,
        font=dict(family="Inter, Malgun Gothic, Apple SD Gothic Neo, sans-serif", color=DARK_TEXT, size=13),
        title=dict(
            font=dict(family="Arial Black, Impact, Malgun Gothic, sans-serif", color=DARK_TEXT, size=20),
            x=0.02,
            xanchor="left",
            pad=dict(t=10, b=20),
        ),
        colorway=PLOTLY_COLORWAY,
        legend=dict(
            bgcolor=RETRO_PANEL,
            bordercolor=DARK_STROKE,
            borderwidth=1,
            font=dict(color=DARK_TEXT),
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
        ),
        hoverlabel=dict(
            bgcolor=RETRO_PANEL,
            bordercolor=DARK_STROKE,
            font=dict(color=DARK_TEXT, family="MS Sans Serif, Tahoma, Malgun Gothic, sans-serif"),
        ),
        margin=dict(l=82, r=108, t=90, b=210),
    )
    fig.update_xaxes(
        color=DARK_TEXT,
        gridcolor=DARK_MUTED,
        zerolinecolor=DARK_STROKE,
        linecolor=DARK_STROKE,
        title_font=dict(color=DARK_TEXT),
        tickfont=dict(color=DARK_TEXT),
        automargin=True,
        title_standoff=26,
    )
    fig.update_yaxes(
        color=DARK_TEXT,
        gridcolor=DARK_MUTED,
        zerolinecolor=DARK_STROKE,
        linecolor=DARK_STROKE,
        title_font=dict(color=DARK_TEXT),
        tickfont=dict(color=DARK_TEXT),
        automargin=True,
        title_standoff=30,
    )
    return fig


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
    max_value = data[metric].max()
    fig.update_traces(
        texttemplate="%{text:,.0f}",
        textposition="inside",
        insidetextanchor="end",
        textfont=dict(color="#F8FAFC", size=12),
        cliponaxis=False,
    )
    fig.update_yaxes(categoryorder="total ascending")
    if pd.notna(max_value) and max_value > 0:
        fig.update_xaxes(range=[0, max_value * 1.32])
    fig.update_layout(
        height=540,
        margin=dict(l=82, r=108, t=90, b=210),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return apply_dark_plotly_theme(fig)


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
    max_value = data[metric].max()
    fig.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    if pd.notna(max_value) and max_value > 0:
        fig.update_yaxes(range=[0, max_value * 1.16])
    fig.update_layout(height=460, margin=dict(l=20, r=35, t=60, b=20))
    return apply_dark_plotly_theme(fig)


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
    return apply_dark_plotly_theme(fig)


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
    return apply_dark_plotly_theme(fig)


def _attach_stop_metadata(hourly_df: pd.DataFrame, stop_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """시간대 데이터에 구·군 같은 정류소 속성이 없으면 정류소 요약표에서 붙입니다."""
    data = hourly_df.copy()
    if stop_df.empty:
        return data

    missing_cols = [col for col in columns if col in stop_df.columns and col not in data.columns]
    empty_cols = [
        col
        for col in columns
        if col in stop_df.columns and col in data.columns and data[col].notna().sum() == 0
    ]
    target_cols = list(dict.fromkeys(missing_cols + empty_cols))
    if not target_cols:
        return data

    if "_merge_key" in data.columns and "_merge_key" in stop_df.columns:
        meta = stop_df[["_merge_key"] + target_cols].dropna(subset=["_merge_key"]).drop_duplicates("_merge_key")
        data = data.drop(columns=[col for col in empty_cols if col in data.columns], errors="ignore")
        data = data.merge(meta, on="_merge_key", how="left")

    unresolved = [col for col in target_cols if col not in data.columns or data[col].notna().sum() == 0]
    if unresolved and "stop_name" in data.columns and "stop_name" in stop_df.columns:
        meta = stop_df[["stop_name"] + unresolved].dropna(subset=["stop_name"]).drop_duplicates("stop_name")
        data = data.drop(columns=[col for col in unresolved if col in data.columns], errors="ignore")
        data = data.merge(meta, on="stop_name", how="left")

    return data


def plot_stop_hour_heatmap(hourly_df: pd.DataFrame, stop_df: pd.DataFrame, metric: str = "boardings", top_n: int = 20):
    """정류소 × 시간대 히트맵을 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    required = {metric, "hour", "stop_name"}
    if hourly_df.empty or not required.issubset(hourly_df.columns):
        return None

    data = hourly_df.copy()
    if not stop_df.empty and metric in stop_df.columns:
        top_stop_df = stop_df.sort_values(metric, ascending=False).head(top_n)
        data_by_key = pd.DataFrame()
        if "_merge_key" in data.columns and "_merge_key" in top_stop_df.columns:
            top_keys = set(top_stop_df["_merge_key"].dropna())
            data_by_key = data[data["_merge_key"].isin(top_keys)].copy()
        if not data_by_key.empty:
            data = data_by_key
        elif "stop_name" in top_stop_df.columns:
            top_names = set(top_stop_df["stop_name"].dropna().astype(str))
            data = data[data["stop_name"].astype(str).isin(top_names)].copy()
    else:
        top_names = (
            data.groupby("stop_name", as_index=False)[metric]
            .sum()
            .sort_values(metric, ascending=False)["stop_name"]
            .head(top_n)
        )
        data = data[data["stop_name"].isin(top_names)].copy()

    if data.empty:
        return None
    pivot = data.pivot_table(index="stop_name", columns="hour", values=metric, aggfunc="sum").fillna(0)
    if pivot.empty:
        return None
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[DARK_SURFACE, ACCENT_BLUE_DEEP, ACCENT_BLUE],
        title=f"정류소 × 시간대 {metric_label(metric)} 히트맵",
        labels=dict(x="시간대", y="정류소", color=metric_label(metric)),
    )
    fig.update_layout(height=560, margin=dict(l=60, r=40, t=70, b=60))
    return apply_dark_plotly_theme(fig)


def plot_district_hour_heatmap(hourly_df: pd.DataFrame, metric: str = "boardings", stop_df: pd.DataFrame | None = None):
    """구·군 × 시간대 히트맵을 만듭니다."""
    px = get_plotly_express()
    if px is None:
        return None
    if hourly_df.empty or metric not in hourly_df.columns or "hour" not in hourly_df.columns:
        return None

    data = hourly_df.copy()
    if "district" not in data.columns or data["district"].notna().sum() == 0:
        data = _attach_stop_metadata(data, stop_df if stop_df is not None else pd.DataFrame(), ["district"])
    if "district" not in data.columns or data["district"].notna().sum() == 0:
        return None

    pivot = data.pivot_table(index="district", columns="hour", values=metric, aggfunc="sum").fillna(0)
    if pivot.empty:
        return None
    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale=[DARK_SURFACE, ACCENT_BLUE_DEEP, ACCENT_BLUE],
        title=f"구·군 × 시간대 {metric_label(metric)} 히트맵",
        labels=dict(x="시간대", y="구·군", color=metric_label(metric)),
    )
    fig.update_layout(height=500, margin=dict(l=60, r=40, t=70, b=60))
    return apply_dark_plotly_theme(fig)


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
    return apply_dark_plotly_theme(fig)


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
    return apply_dark_plotly_theme(fig)


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
    hover_data = {
        "stop_name": True,
        "boardings": ":,",
        "route_count": ":,",
        "boardings_per_route": ":,.1f",
    }
    for optional_col in ["district", "peak_hour_label", "stop_type"]:
        if optional_col in data.columns:
            hover_data[optional_col] = True
    fig = px.scatter(
        data,
        x="route_count",
        y="boardings",
        size="boardings_per_route",
        color="stop_type" if "stop_type" in data.columns else None,
        hover_data=hover_data,
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
        fig.add_vline(x=route_line, line_dash="dash", line_color=ACCENT_BLUE)
    if pd.notna(demand_line):
        fig.add_hline(y=demand_line, line_dash="dash", line_color=ACCENT_BLUE)
    if pd.notna(route_line) and pd.notna(demand_line):
        fig.add_shape(
            type="rect",
            x0=0,
            x1=route_line,
            y0=demand_line,
            y1=data["boardings"].max(),
            fillcolor="rgba(251, 113, 133, 0.14)",
            line_width=0,
            layer="below",
        )
        fig.add_annotation(
            x=route_line / 2 if route_line else 0,
            y=data["boardings"].max() * 0.96,
            text="추가 검토 후보 영역<br>노선 수는 적고 승차 수요는 높음",
            showarrow=False,
            align="left",
            bgcolor=RETRO_PANEL,
            bordercolor=DARK_STROKE,
            borderwidth=1,
            font=dict(color=DARK_TEXT, size=12),
        )
        fig.add_annotation(
            x=route_line,
            y=data["boardings"].min(),
            text=f"노선 {route_line:,.0f}개 이하",
            showarrow=False,
            yshift=-34,
            bgcolor=RETRO_PANEL,
            bordercolor=DARK_STROKE,
            borderwidth=1,
            font=dict(color=DARK_TEXT, size=11),
        )
        fig.add_annotation(
            x=data["route_count"].max(),
            y=demand_line,
            text=f"승차 {demand_line:,.0f}명 이상",
            showarrow=False,
            xanchor="right",
            yshift=12,
            bgcolor=RETRO_PANEL,
            bordercolor=DARK_STROKE,
            borderwidth=1,
            font=dict(color=DARK_TEXT, size=11),
        )
    fig.update_layout(height=600, margin=dict(l=20, r=20, t=60, b=20))
    return apply_dark_plotly_theme(fig)


def bus_stop_icon_data_uri(pin_color: str = "#2563EB") -> str:
    """지도 위에 표시할 버스 정류장 핀 SVG를 데이터 URI로 만듭니다."""
    svg = """
    <svg xmlns="http://www.w3.org/2000/svg" width="128" height="128" viewBox="0 0 128 128">
      <path d="M64 124C54 110 27 78 27 50C27 29 44 12 64 12C84 12 101 29 101 50C101 78 74 110 64 124Z" fill="{pin_color}" stroke="#111111" stroke-width="2.5"/>
      <path d="M84 91C78 101 71 112 64 124C54 110 27 78 27 50C27 48 27 46 28 44C39 75 63 87 84 91Z" fill="#000000" opacity="0.16"/>
      <circle cx="64" cy="50" r="35" fill="#F8F8F8" stroke="#111111" stroke-width="1.8"/>
      <path d="M47 35H81C87 35 91 39 91 45V72C91 78 87 82 81 82H47C41 82 37 78 37 72V45C37 39 41 35 47 35Z" fill="#111111"/>
      <path d="M50 44H78C81 44 83 46 83 49V63H45V49C45 46 47 44 50 44Z" fill="#F8F8F8"/>
      <circle cx="50" cy="72" r="6" fill="#F8F8F8"/>
      <circle cx="78" cy="72" r="6" fill="#F8F8F8"/>
      <rect x="42" y="82" width="12" height="12" fill="#111111"/>
      <rect x="74" y="82" width="12" height="12" fill="#111111"/>
      <rect x="29" y="51" width="9" height="5" fill="#111111"/>
      <rect x="90" y="51" width="9" height="5" fill="#111111"/>
    </svg>
    """
    return "data:image/svg+xml;charset=utf-8," + quote(" ".join(svg.format(pin_color=pin_color).split()))


def map_density_style(value: float, high_cutoff: float, very_high_cutoff: float) -> tuple[str, str]:
    """노선당 승차 인원 수준을 3단계 색상과 라벨로 바꿉니다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if pd.isna(number):
        number = 0.0
    if number >= very_high_cutoff:
        return "#FF4048", "매우 높음"
    if number >= high_cutoff:
        return "#FFD400", "높음"
    return "#2563EB", "보통"


def map_size_style(value: float, medium_cutoff: float, large_cutoff: float, huge_cutoff: float) -> tuple[int, str]:
    """전체 승차 인원을 지도 아이콘 크기 4단계로 바꿉니다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if pd.isna(number):
        number = 0.0
    if number >= huge_cutoff:
        return 42, "매우 큼"
    if number >= large_cutoff:
        return 26, "큼"
    if number >= medium_cutoff:
        return 16, "보통"
    return 10, "작음"


def create_pydeck_map(stop_df: pd.DataFrame, max_icons: int | None = 900):
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
    if max_icons is not None and max_icons > 0 and len(data) > max_icons:
        data = data.sort_values("boardings", ascending=False).head(max_icons).copy()
    per_route = data.get("boardings_per_route", pd.Series(0, index=data.index)).fillna(0)
    boardings = data["boardings"].fillna(0).clip(lower=0)
    boardings_base = boardings.quantile(0.95)
    if pd.isna(boardings_base) or boardings_base <= 0:
        boardings_base = boardings.max() if boardings.max() > 0 else 1
    route_base = per_route.quantile(0.95)
    if pd.isna(route_base) or route_base <= 0:
        route_base = per_route.max() if per_route.max() > 0 else 1

    # 원형 점 대신 버스 정류장 핀 아이콘을 사용합니다.
    # 연속값 크기는 지도에서 차이가 약하게 보여서, 발표 화면에서도 구분되는 4단계 크기로 나눕니다.
    medium_cutoff = boardings.quantile(0.50)
    large_cutoff = boardings.quantile(0.80)
    huge_cutoff = boardings.quantile(0.95)
    if pd.isna(medium_cutoff):
        medium_cutoff = 0
    if pd.isna(large_cutoff) or large_cutoff <= medium_cutoff:
        large_cutoff = medium_cutoff + 1
    if pd.isna(huge_cutoff) or huge_cutoff <= large_cutoff:
        huge_cutoff = large_cutoff + 1
    size_styles = [map_size_style(value, medium_cutoff, large_cutoff, huge_cutoff) for value in boardings]
    data["icon_size"] = [item[0] for item in size_styles]
    data["icon_size_level"] = [item[1] for item in size_styles]
    high_cutoff = per_route.quantile(0.70)
    very_high_cutoff = per_route.quantile(0.90)
    if pd.isna(high_cutoff):
        high_cutoff = 0
    if pd.isna(very_high_cutoff) or very_high_cutoff <= high_cutoff:
        very_high_cutoff = per_route.max() if per_route.max() > 0 else high_cutoff + 1
    styles = [map_density_style(value, high_cutoff, very_high_cutoff) for value in per_route]
    data["density_color"] = [item[0] for item in styles]
    data["density_level"] = [item[1] for item in styles]
    icon_cache = {
        "#2563EB": {"url": bus_stop_icon_data_uri("#2563EB"), "width": 128, "height": 128, "anchorY": 128, "mask": False},
        "#FFD400": {"url": bus_stop_icon_data_uri("#FFD400"), "width": 128, "height": 128, "anchorY": 128, "mask": False},
        "#FF4048": {"url": bus_stop_icon_data_uri("#FF4048"), "width": 128, "height": 128, "anchorY": 128, "mask": False},
    }
    data["icon_data"] = [icon_cache[color] for color in data["density_color"]]

    center_lat = float(data["lat"].mean())
    center_lon = float(data["lon"].mean())
    layer = pdk.Layer(
        "IconLayer",
        data=data,
        get_position="[lon, lat]",
        get_icon="icon_data",
        get_size="icon_size",
        size_units="pixels",
        size_scale=1,
        size_min_pixels=8,
        size_max_pixels=46,
        pickable=True,
        auto_highlight=True,
    )
    tooltip = {
        "html": """
        <b>{stop_name}</b><br/>
        구·군: {district}<br/>
        승차 인원: {boardings}<br/>
        하차 인원: {alightings}<br/>
        경유 노선 수: {route_count}<br/>
        노선당 승차 인원: {boardings_per_route}<br/>
        노선당 승차 밀도: {density_level}<br/>
        아이콘 크기: {icon_size_level}<br/>
        정류소 유형: {stop_type}
        """,
        "style": {
            "backgroundColor": RETRO_PANEL,
            "color": DARK_TEXT,
            "border": f"2px solid {DARK_STROKE}",
            "borderRadius": "0px",
            "boxShadow": "inset -1px -1px 0 #808080, inset 1px 1px 0 #ffffff",
            "fontFamily": "MS Sans Serif, Malgun Gothic, sans-serif",
            "padding": "12px",
        },
    }
    view_state = pdk.ViewState(latitude=center_lat, longitude=center_lon, zoom=11, pitch=0)
    return pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
    )


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
        yaxis=dict(title="승차 인원", automargin=True),
        yaxis2=dict(
            title=dict(text=weather_label, standoff=18),
            overlaying="y",
            side="right",
            automargin=True,
        ),
        height=540,
        margin=dict(l=82, r=108, t=90, b=210),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    )
    return apply_dark_plotly_theme(fig)


def plot_weather_monthly_line(data: pd.DataFrame, y_col: str, y_label: str, title: str):
    """월별 날씨 또는 버스 지표의 추세 선그래프를 만듭니다."""
    px = get_plotly_express()
    if px is None or data.empty or y_col not in data.columns:
        return None
    fig = px.line(data, x="period", y=y_col, markers=True, title=title, labels={"period": "연도·월", y_col: y_label})
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return apply_dark_plotly_theme(fig)


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
    return apply_dark_plotly_theme(fig)


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
    return apply_dark_plotly_theme(fig)


def plot_cluster_pattern_heatmap(patterns: pd.DataFrame):
    """K-means 군집별 시간대 이용 비율을 퍼센트 히트맵으로 보여줍니다."""
    px = get_plotly_express()
    if px is None or patterns.empty:
        return None
    hour_cols = sorted([col for col in patterns.columns if isinstance(col, (int, float))], key=int)
    if not hour_cols:
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
        color_continuous_scale=[DARK_SURFACE, ACCENT_BLUE_DEEP, ACCENT_BLUE],
        title="군집별 시간대 승차 비율",
        labels=dict(x="시간대", y="군집 유형", color="승차 비율(%)"),
    )
    fig.update_traces(
        texttemplate="%{z:.1f}%",
        hovertemplate="군집: %{y}<br>시간대: %{x}<br>승차 비율: %{z:.1f}%<extra></extra>",
    )
    fig.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=20))
    return apply_dark_plotly_theme(fig)


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
        color_continuous_scale=[ACCENT_BLUE_DEEP, DARK_SURFACE, ALERT_ROSE],
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
    plt.savefig(path, dpi=300, bbox_inches="tight", facecolor=DARK_BG)
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
    sns.set_theme(
        style="darkgrid",
        rc={
            "axes.facecolor": DARK_SURFACE,
            "figure.facecolor": DARK_BG,
            "axes.edgecolor": DARK_STROKE,
            "axes.labelcolor": DARK_TEXT,
            "xtick.color": DARK_MUTED,
            "ytick.color": DARK_MUTED,
            "text.color": DARK_TEXT,
            "grid.color": "#26272C",
        },
    )
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
