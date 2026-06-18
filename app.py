from pathlib import Path
from html import escape
import importlib

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

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

import src.analysis as analysis_module


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

SOFT_BG = "#0A0A0B"
SOFT_SURFACE = "#141416"
SOFT_SURFACE_RAISED = "#1A1B1F"
SOFT_TEXT = "#F5F5F5"
SOFT_MUTED = "#8A8A8F"
SOFT_ACCENT = "#89AACC"
SOFT_ACCENT_LIGHT = "#4E85BF"
SOFT_ACCENT_GRADIENT = "linear-gradient(90deg, #89AACC 0%, #4E85BF 100%)"
SOFT_TEAL = "#4ADE80"
SOFT_ROSE = "#FB7185"
SOFT_STROKE = "#26272C"
SOFT_SHADOW_LIGHT = "rgba(137, 170, 204, 0.12)"
SOFT_SHADOW_DARK = "rgba(0, 0, 0, 0.50)"
SOFT_SHADOW = (
    "0 22px 70px rgba(0, 0, 0, 0.44), "
    "inset 0 1px 0 rgba(255, 255, 255, 0.04)"
)
SOFT_SHADOW_HOVER = (
    "0 26px 90px rgba(0, 0, 0, 0.58), "
    "0 0 0 1px rgba(137, 170, 204, 0.34), "
    "0 0 34px rgba(78, 133, 191, 0.14)"
)
SOFT_SHADOW_INSET = (
    "inset 0 1px 0 rgba(255, 255, 255, 0.04), "
    "inset 0 0 0 1px rgba(255, 255, 255, 0.05)"
)
SOFT_SHADOW_INSET_DEEP = (
    "inset 0 0 0 1px rgba(137, 170, 204, 0.20), "
    "inset 0 -24px 60px rgba(0, 0, 0, 0.35)"
)


def apply_neumorphic_theme() -> None:
    """Streamlit 기본 UI 위에 프리미엄 다크 대시보드 디자인 토큰을 적용합니다."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@1&family=Inter:wght@300;400;500;600;700;800&display=swap');

        :root {{
            --soft-bg: {SOFT_BG};
            --soft-surface: {SOFT_SURFACE};
            --soft-surface-raised: {SOFT_SURFACE_RAISED};
            --soft-text: {SOFT_TEXT};
            --soft-muted: {SOFT_MUTED};
            --soft-accent: {SOFT_ACCENT};
            --soft-accent-light: {SOFT_ACCENT_LIGHT};
            --soft-teal: {SOFT_TEAL};
            --soft-rose: {SOFT_ROSE};
            --soft-stroke: {SOFT_STROKE};
            --accent-gradient: {SOFT_ACCENT_GRADIENT};
            --soft-shadow: {SOFT_SHADOW};
            --soft-shadow-hover: {SOFT_SHADOW_HOVER};
            --soft-inset: {SOFT_SHADOW_INSET};
            --soft-inset-deep: {SOFT_SHADOW_INSET_DEEP};
        }}

        html, body, [class*="css"] {{
            font-family: "Inter", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
            color: var(--soft-text);
        }}

        .stApp {{
            background:
                radial-gradient(circle at 12% 0%, rgba(137, 170, 204, 0.15), transparent 32rem),
                radial-gradient(circle at 88% 8%, rgba(78, 133, 191, 0.14), transparent 28rem),
                linear-gradient(90deg, rgba(255,255,255,0.018) 1px, transparent 1px),
                linear-gradient(180deg, rgba(255,255,255,0.012) 1px, transparent 1px),
                linear-gradient(180deg, #050506 0%, var(--soft-bg) 48%, #050506 100%);
            background-size: auto, auto, 72px 72px, 72px 72px, auto;
            color: var(--soft-text);
        }}

        .block-container {{
            max-width: 1480px;
            padding-top: 2rem;
            padding-bottom: 5rem;
        }}

        h1, h2, h3, h4, h5, h6 {{
            font-family: "Inter", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
            color: var(--soft-text);
            letter-spacing: 0;
        }}

        h1 {{
            font-size: clamp(1.9rem, 2.6vw, 3rem);
            line-height: 1.18;
            font-weight: 800;
            margin-bottom: 0.65rem;
        }}

        h2, h3 {{
            font-weight: 800;
        }}

        p, li, label, span {{
            color: inherit;
        }}

        .app-hero {{
            position: relative;
            margin: 0.25rem 0 1.8rem;
            padding: clamp(1.65rem, 3vw, 3rem);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 30px;
            background:
                linear-gradient(135deg, rgba(137, 170, 204, 0.13), transparent 38%),
                radial-gradient(circle at 88% 0%, rgba(78, 133, 191, 0.22), transparent 24rem),
                rgba(20, 20, 22, 0.78);
            box-shadow: var(--soft-shadow);
            overflow: hidden;
        }}

        .app-hero::before {{
            content: "";
            position: absolute;
            inset: 0;
            pointer-events: none;
            background-image: radial-gradient(circle, rgba(255,255,255,0.18) 1px, transparent 1px);
            background-size: 5px 5px;
            opacity: 0.08;
            mix-blend-mode: screen;
        }}

        .app-hero__eyebrow {{
            position: relative;
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            margin-bottom: 0.9rem;
            padding: 0.36rem 0.82rem;
            border: 1px solid rgba(137, 170, 204, 0.26);
            border-radius: 999px;
            color: var(--soft-text);
            background: rgba(255, 255, 255, 0.035);
            box-shadow: var(--soft-inset);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.28em;
            text-transform: uppercase;
        }}

        .app-hero__title {{
            position: relative;
            max-width: 1180px;
            font-family: "Instrument Serif", "Inter", "Malgun Gothic", serif;
            color: var(--soft-text);
            font-size: clamp(1.95rem, 3.25vw, 3.65rem);
            font-style: italic;
            font-weight: 400;
            line-height: 1.12;
            margin: 0 0 1rem;
            text-wrap: pretty;
            word-break: keep-all;
            overflow-wrap: normal;
        }}

        .app-hero__body {{
            position: relative;
            max-width: 880px;
            color: var(--soft-muted);
            font-size: 1.02rem;
            line-height: 1.85;
            margin: 0;
        }}

        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)),
                rgba(10, 10, 11, 0.98);
            border-right: 1px solid var(--soft-stroke);
            box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.03), 18px 0 80px rgba(0,0,0,0.30);
        }}

        section[data-testid="stSidebar"] > div {{
            background: transparent;
        }}

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: var(--soft-text);
        }}

        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
            padding-top: 1.4rem;
        }}

        section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        section[data-testid="stSidebar"] label {{
            color: var(--soft-muted);
            font-size: 0.88rem;
            font-weight: 700;
        }}

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div,
        section[data-testid="stSidebar"] div[data-baseweb="input"] > div {{
            background: rgba(255,255,255,0.045) !important;
        }}

        div[data-testid="stMetric"] {{
            min-height: 132px;
            padding: 1.05rem 1.05rem 1rem;
            border: 1px solid rgba(137, 170, 204, 0.20);
            border-radius: 22px;
            background: linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018));
            box-shadow: var(--soft-shadow);
            transition: transform 300ms ease-out, box-shadow 300ms ease-out;
        }}

        div[data-testid="stMetric"]:hover {{
            transform: translateY(-1px);
            box-shadow: var(--soft-shadow-hover);
        }}

        div[data-testid="stMetricLabel"] p {{
            color: var(--soft-muted);
            font-size: 0.82rem;
            font-weight: 800;
            line-height: 1.35;
        }}

        div[data-testid="stMetricValue"] {{
            color: var(--soft-text);
            font-family: "Inter", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
            font-size: clamp(1.15rem, 1.35vw, 1.65rem);
            font-style: normal;
            font-weight: 800;
            line-height: 1.28;
            overflow-wrap: anywhere;
            font-variant-numeric: tabular-nums;
        }}

        div[data-testid="stMetricDelta"] {{
            color: var(--soft-accent);
            font-weight: 800;
        }}

        div[data-testid="stTabs"] button {{
            min-height: 46px;
            border-radius: 999px;
            color: var(--soft-muted);
            background: transparent;
            transition: all 300ms ease-out;
        }}

        div[data-testid="stTabs"] button:hover {{
            color: var(--soft-text);
            background: rgba(255, 255, 255, 0.045);
        }}

        div[data-testid="stTabs"] button[aria-selected="true"] {{
            color: var(--soft-text);
            background: linear-gradient(90deg, rgba(137,170,204,0.22), rgba(78,133,191,0.18));
            box-shadow: inset 0 0 0 1px rgba(137,170,204,0.28);
            font-weight: 700;
        }}

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
            width: fit-content;
            max-width: 100%;
            gap: 0.5rem;
            padding: 0.45rem;
            border: 1px solid var(--soft-stroke);
            border-radius: 999px;
            background: rgba(20, 20, 22, 0.88);
            box-shadow: var(--soft-inset);
            overflow-x: auto;
        }}

        button[kind], div[data-testid="stDownloadButton"] button {{
            min-height: 44px;
            border: 1px solid rgba(137, 170, 204, 0.25) !important;
            border-radius: 999px !important;
            color: var(--soft-text) !important;
            background: linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.025)) !important;
            box-shadow: var(--soft-shadow) !important;
            transition: transform 300ms ease-out, box-shadow 300ms ease-out !important;
        }}

        button[kind]:hover, div[data-testid="stDownloadButton"] button:hover {{
            transform: translateY(-1px);
            color: var(--soft-text) !important;
            box-shadow: var(--soft-shadow-hover) !important;
        }}

        button[kind]:active, div[data-testid="stDownloadButton"] button:active {{
            transform: translateY(0.5px);
            box-shadow: var(--soft-inset) !important;
        }}

        button:focus, input:focus, textarea:focus, [role="button"]:focus-visible {{
            outline: 2px solid var(--soft-accent) !important;
            outline-offset: 2px !important;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="base-input"],
        div[data-testid="stNumberInput"] input {{
            border: 1px solid var(--soft-stroke) !important;
            border-radius: 14px !important;
            background: rgba(255, 255, 255, 0.035) !important;
            color: var(--soft-text) !important;
            box-shadow: var(--soft-inset) !important;
        }}

        div[data-baseweb="select"] span,
        div[data-baseweb="base-input"] input {{
            color: var(--soft-text) !important;
        }}

        div[data-testid="stSlider"] [data-baseweb="slider"] > div {{
            color: var(--soft-accent);
        }}

        div[data-testid="stCheckbox"] label {{
            min-height: 44px;
            align-items: center;
        }}

        div[data-testid="stSlider"] [role="slider"] {{
            background: var(--accent-gradient) !important;
            border-color: rgba(255,255,255,0.24) !important;
        }}

        div[data-testid="stAlert"],
        details[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stPydeckChart"],
        div[data-testid="stVegaLiteChart"] {{
            border: 1px solid var(--soft-stroke) !important;
            border-radius: 26px !important;
            background: rgba(20, 20, 22, 0.82) !important;
            box-shadow: var(--soft-shadow);
            overflow: hidden;
        }}

        div[data-testid="stAlert"] {{
            padding: 0.2rem;
        }}

        div[data-testid="stPlotlyChart"],
        div[data-testid="stPydeckChart"],
        div[data-testid="stVegaLiteChart"] {{
            padding: 0.7rem;
            margin-bottom: 1.1rem;
        }}

        div[data-testid="stPlotlyChart"]:nth-of-type(even) {{
            margin-top: 0.65rem;
        }}

        div[data-testid="stDataFrame"] {{
            padding: 0.35rem;
        }}

        div[data-testid="stDataFrame"] canvas,
        div[data-testid="stDataFrame"] [role="gridcell"],
        div[data-testid="stDataFrame"] [role="columnheader"] {{
            font-family: "Inter", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif !important;
            font-variant-numeric: tabular-nums;
        }}

        div[data-testid="stDataFrame"] [role="grid"] {{
            border-radius: 20px;
            overflow: hidden;
            color: var(--soft-text);
        }}

        details[data-testid="stExpander"] > summary {{
            color: var(--soft-text);
            font-weight: 800;
        }}

        .soft-note {{
            margin: 0.4rem 0 1rem;
            padding: 1rem 1.2rem;
            border: 1px solid var(--soft-stroke);
            border-radius: 20px;
            color: var(--soft-muted);
            background: rgba(255,255,255,0.035);
            box-shadow: var(--soft-inset);
            line-height: 1.65;
            font-weight: 500;
        }}

        .section-kicker {{
            position: relative;
            margin: 1.6rem 0 1rem;
            padding: 1rem 0 0.35rem;
        }}

        .section-kicker::before {{
            content: "";
            display: block;
            width: 2.5rem;
            height: 1px;
            margin-bottom: 0.75rem;
            background: var(--accent-gradient);
            box-shadow: 0 0 12px rgba(137,170,204,0.35);
        }}

        .section-kicker__eyebrow {{
            margin-bottom: 0.45rem;
            color: var(--soft-muted);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.28em;
            text-transform: uppercase;
        }}

        .section-kicker__title {{
            margin: 0;
            color: var(--soft-text);
            font-family: "Inter", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
            font-size: clamp(1.65rem, 2.15vw, 2.6rem);
            font-style: normal;
            font-weight: 850;
            line-height: 1.2;
        }}

        .section-kicker__body {{
            max-width: 760px;
            margin-top: 0.7rem;
            color: var(--soft-muted);
            line-height: 1.7;
        }}

        .soft-pill {{
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.25rem 0.75rem;
            border: 1px solid rgba(137, 170, 204, 0.24);
            border-radius: 999px;
            color: var(--soft-text);
            background: rgba(255,255,255,0.04);
            box-shadow: var(--soft-inset);
            font-weight: 700;
        }}

        .soft-metric-card {{
            min-height: 124px;
            position: relative;
            padding: 1rem 1.05rem 1.15rem;
            border: 1px solid rgba(137, 170, 204, 0.22);
            border-radius: 22px;
            background:
                linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.020)),
                rgba(20, 20, 22, 0.84);
            box-shadow: var(--soft-shadow);
            overflow: hidden;
            transition: transform 300ms ease-out, box-shadow 300ms ease-out;
        }}

        .soft-metric-card::before {{
            display: none;
        }}

        .soft-metric-card::after {{
            content: "";
            position: absolute;
            left: 1rem;
            right: 1rem;
            bottom: 0;
            height: 3px;
            border-radius: 999px;
            background: var(--accent-gradient);
            opacity: 0.85;
            box-shadow: 0 0 8px rgba(137,170,204,0.35);
        }}

        .soft-metric-card:hover {{
            transform: translateY(-1px);
            box-shadow: var(--soft-shadow-hover);
        }}

        .soft-metric-card__label {{
            color: var(--soft-muted);
            font-size: 0.74rem;
            font-weight: 750;
            line-height: 1.35;
            margin-bottom: 0.55rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}

        .soft-metric-card__value {{
            color: var(--soft-text);
            font-family: "Inter", "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
            font-size: clamp(1.18rem, 1.45vw, 1.8rem);
            font-style: normal;
            font-weight: 800;
            line-height: 1.22;
            overflow-wrap: anywhere;
            word-break: break-word;
            font-variant-numeric: tabular-nums;
            max-width: 100%;
        }}

        div[data-testid="stHorizontalBlock"] > div:nth-child(even) div[data-testid="stPlotlyChart"] {{
            margin-top: 1.15rem;
        }}

        div[data-testid="stHorizontalBlock"] > div:nth-child(odd) div[data-testid="stPlotlyChart"] {{
            border-color: rgba(137,170,204,0.24) !important;
        }}

        .soft-metric-card__delta {{
            display: inline-flex;
            margin-top: 0.65rem;
            padding: 0.22rem 0.58rem;
            border: 1px solid rgba(137, 170, 204, 0.22);
            border-radius: 999px;
            color: var(--soft-accent);
            background: rgba(137, 170, 204, 0.08);
            box-shadow: var(--soft-inset);
            font-size: 0.78rem;
            font-weight: 700;
        }}

        .stMarkdown a {{
            color: var(--soft-accent);
            font-weight: 800;
        }}

        hr {{
            border: 0;
            height: 1px;
            background: rgba(255, 255, 255, 0.10);
        }}

        .cosmic-footer {{
            position: relative;
            margin-top: 3rem;
            padding: 2.4rem 0 0.4rem;
            border-top: 1px solid var(--soft-stroke);
            overflow: hidden;
        }}

        .cosmic-marquee {{
            display: flex;
            width: max-content;
            animation: cosmic-marquee 38s linear infinite;
            color: rgba(245, 245, 245, 0.92);
            font-family: "Instrument Serif", "Inter", "Malgun Gothic", serif;
            font-size: clamp(2rem, 4.8vw, 5rem);
            font-style: italic;
            line-height: 0.95;
            white-space: nowrap;
        }}

        .cosmic-marquee span {{
            padding-right: 2rem;
            background: var(--accent-gradient);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }}

        .cosmic-footer__meta {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.8rem;
            margin-top: 1.3rem;
            color: var(--soft-muted);
            font-size: 0.86rem;
        }}

        .cosmic-dot {{
            display: inline-block;
            width: 0.55rem;
            height: 0.55rem;
            margin-right: 0.45rem;
            border-radius: 999px;
            background: #4ade80;
            box-shadow: 0 0 0 rgba(74, 222, 128, 0.55);
            animation: pulse-dot 1.8s ease-in-out infinite;
        }}

        @keyframes cosmic-marquee {{
            from {{ transform: translateX(0); }}
            to {{ transform: translateX(-50%); }}
        }}

        @keyframes pulse-dot {{
            0%, 100% {{ box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.45); }}
            50% {{ box-shadow: 0 0 0 8px rgba(74, 222, 128, 0); }}
        }}

        @media (max-width: 900px) {{
            .block-container {{
                padding-left: 1rem;
                padding-right: 1rem;
            }}
            .app-hero {{
                padding: 1.45rem;
                border-radius: 28px;
            }}
            div[data-testid="stMetric"] {{
                min-height: 108px;
            }}
            .soft-metric-card {{
                min-height: 108px;
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_retro_90s_overrides() -> None:
    """90년대 Windows/초기 웹 느낌의 최종 CSS 오버라이드를 적용합니다."""
    st.markdown(
        """
        <style>
        :root {
            --retro-bg: #c0c0c0;
            --retro-bg-dark: #a8a8a8;
            --retro-text: #000000;
            --retro-muted: #808080;
            --retro-blue: #000080;
            --retro-blue-bright: #0000ff;
            --retro-blue-end: #1084d0;
            --retro-red: #ff0000;
            --retro-yellow: #ffff00;
            --retro-panel: #ffffcc;
            --retro-white: #ffffff;
            --retro-green: #00aa00;
            --retro-border-dark: #808080;
            --retro-border-darker: #404040;
            --retro-highlight: #dfdfdf;
        }

        html, body, [class*="css"] {
            font-family: "MS Sans Serif", "Segoe UI", Tahoma, Geneva, Verdana, sans-serif !important;
            color: var(--retro-text) !important;
        }

        .stApp {
            background-color: var(--retro-bg) !important;
            background-image:
                linear-gradient(45deg, #b8b8b8 25%, transparent 25%),
                linear-gradient(-45deg, #b8b8b8 25%, transparent 25%),
                linear-gradient(45deg, transparent 75%, #b8b8b8 75%),
                linear-gradient(-45deg, transparent 75%, #b8b8b8 75%) !important;
            background-size: 4px 4px !important;
            background-position: 0 0, 0 2px, 2px -2px, -2px 0 !important;
        }

        html * {
            border-radius: 0 !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }

        html,
        body,
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: hidden !important;
        }

        .block-container {
            max-width: none !important;
            width: 100% !important;
            box-sizing: border-box !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1.2rem !important;
            padding-bottom: 2.2rem !important;
        }

        div[data-testid="stHorizontalBlock"] {
            width: 100% !important;
            align-items: stretch !important;
        }

        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            min-width: 0 !important;
        }

        h1, h2, h3, h4, h5, h6,
        .section-kicker__title,
        .app-hero__title {
            font-family: "Arial Black", Impact, Haettenschweiler, "Malgun Gothic", sans-serif !important;
            color: var(--retro-text) !important;
            font-style: normal !important;
            font-weight: 900 !important;
            letter-spacing: 0 !important;
            line-height: 1.15 !important;
            text-shadow: 2px 2px 0 #808080;
        }

        .app-hero {
            margin: 0 0 1rem !important;
            padding: 2.35rem 1rem 1rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: var(--retro-bg) !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
            overflow: hidden !important;
            position: relative !important;
        }

        .app-hero::before {
            content: "" !important;
            position: absolute !important;
            left: 0 !important;
            right: 0 !important;
            top: 0 !important;
            height: 1.75rem !important;
            background: linear-gradient(90deg, #000080, #1084d0) !important;
            opacity: 1 !important;
            pointer-events: none !important;
        }

        .app-hero::after {
            content: "DAEGU_BUS_DASHBOARD.EXE" !important;
            position: absolute !important;
            left: 0.5rem !important;
            top: 0.22rem !important;
            color: #ffffff !important;
            font-family: "MS Sans Serif", Tahoma, sans-serif !important;
            font-size: 0.82rem !important;
            font-weight: 800 !important;
            text-shadow: none !important;
            z-index: 2 !important;
        }

        .app-hero__eyebrow {
            display: inline-block !important;
            margin: 0 0 0.7rem !important;
            padding: 0.22rem 0.45rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #ffff00 !important;
            color: #000000 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
            font-family: "Courier New", Courier, monospace !important;
            font-size: 0.78rem !important;
            letter-spacing: 0 !important;
            text-transform: uppercase !important;
        }

        .app-hero__title {
            max-width: none !important;
            margin: 0 0 0.8rem !important;
            font-size: clamp(1.7rem, 2.7vw, 3rem) !important;
            word-break: keep-all !important;
            white-space: nowrap !important;
        }

        .app-hero__body,
        .section-kicker__body,
        p, li {
            color: #000000 !important;
            font-size: 0.96rem !important;
            line-height: 1.55 !important;
            text-shadow: none !important;
        }

        .section-kicker {
            margin: 1rem 0 0.7rem !important;
            padding: 0.6rem !important;
            border: 2px solid !important;
            border-color: #808080 #ffffff #ffffff #808080 !important;
            background: var(--retro-panel) !important;
            box-shadow: inset 1px 1px 0 #404040, inset -1px -1px 0 #dfdfdf !important;
        }

        .section-kicker::before {
            height: 4px !important;
            width: 100% !important;
            margin-bottom: 0.55rem !important;
            background: linear-gradient(to bottom, #808080 0%, #808080 50%, #ffffff 50%, #ffffff 100%) !important;
            box-shadow: none !important;
        }

        .section-kicker__eyebrow {
            display: inline-block !important;
            margin-bottom: 0.35rem !important;
            padding: 0.15rem 0.35rem !important;
            background: #ff0000 !important;
            color: #ffffff !important;
            font-family: "Courier New", Courier, monospace !important;
            font-size: 0.72rem !important;
            font-weight: 800 !important;
            letter-spacing: 0 !important;
            text-transform: uppercase !important;
            animation: retro-blink 1s step-end infinite;
        }

        .section-kicker__title {
            font-size: clamp(1.35rem, 2.4vw, 2rem) !important;
            text-transform: uppercase !important;
        }

        section[data-testid="stSidebar"] {
            background: #c0c0c0 !important;
            border-right: 2px solid #000000 !important;
            box-shadow: inset -1px 0 0 #808080 !important;
        }

        section[data-testid="stSidebar"] * {
            color: #000000 !important;
            text-shadow: none !important;
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-family: "Arial Black", Impact, "Malgun Gothic", sans-serif !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div,
        div[data-baseweb="base-input"],
        div[data-testid="stNumberInput"] input {
            border: 2px solid !important;
            border-color: #808080 #ffffff #ffffff #808080 !important;
            background: #ffffff !important;
            color: #000000 !important;
            box-shadow: inset 1px 1px 0 #404040, inset -1px -1px 0 #dfdfdf !important;
            font-family: "MS Sans Serif", Tahoma, sans-serif !important;
        }

        button[kind],
        div[data-testid="stDownloadButton"] button,
        div[data-testid="stButton"] button {
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            color: #000000 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
            font-family: "MS Sans Serif", Tahoma, sans-serif !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
        }

        div[data-testid="stDownloadButton"] {
            display: inline-block !important;
            width: auto !important;
            max-width: 100% !important;
            margin: 0.85rem 0 1.15rem !important;
        }

        div[data-testid="stDownloadButton"] button {
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            width: auto !important;
            min-width: 240px !important;
            max-width: 100% !important;
            min-height: 42px !important;
            height: 42px !important;
            padding: 0.45rem 1rem !important;
            white-space: nowrap !important;
            word-break: keep-all !important;
            writing-mode: horizontal-tb !important;
            line-height: 1.2 !important;
        }

        div[data-testid="stDownloadButton"] button *,
        div[data-testid="stDownloadButton"] button p,
        div[data-testid="stDownloadButton"] button span {
            white-space: nowrap !important;
            word-break: keep-all !important;
            writing-mode: horizontal-tb !important;
            line-height: 1.2 !important;
        }

        button[kind]:active,
        div[data-testid="stDownloadButton"] button:active,
        div[data-testid="stButton"] button:active {
            border-color: #808080 #ffffff #ffffff #808080 !important;
            box-shadow: inset 1px 1px 0 #404040, inset -1px -1px 0 #dfdfdf !important;
            transform: translate(1px, 1px) !important;
        }

        button:focus,
        input:focus,
        textarea:focus,
        [role="button"]:focus-visible {
            outline: 2px dotted #000000 !important;
            outline-offset: 2px !important;
        }

        button[kind="icon"],
        button[kind="headerNoPadding"],
        button[data-testid="stBaseButton-headerNoPadding"],
        div[data-testid="stTooltipHoverTarget"] button {
            min-width: 1.1rem !important;
            width: 1.1rem !important;
            height: 1.1rem !important;
            padding: 0 !important;
            border: 0 !important;
            background: transparent !important;
            box-shadow: none !important;
            color: #404040 !important;
            text-transform: none !important;
        }

        button[kind="icon"] svg,
        button[kind="headerNoPadding"] svg,
        button[data-testid="stBaseButton-headerNoPadding"] svg,
        div[data-testid="stTooltipHoverTarget"] button svg {
            width: 0.95rem !important;
            height: 0.95rem !important;
        }

        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            display: flex !important;
            width: 100% !important;
            box-sizing: border-box !important;
            padding: 0.7rem !important;
            gap: 0.5rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
            overflow: hidden !important;
        }

        div[data-testid="stTabs"] [role="tab"] {
            flex: 1 1 0 !important;
            width: 0 !important;
            min-width: 0 !important;
            min-height: 54px !important;
            padding: 0.55rem 0.3rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            color: #000000 !important;
            font-weight: 800 !important;
            white-space: normal !important;
            justify-content: center !important;
            text-align: center !important;
        }

        div[data-testid="stTabs"] [role="tab"] p {
            width: 100% !important;
            font-size: clamp(0.68rem, 0.72vw, 0.9rem) !important;
            line-height: 1.15 !important;
            margin: 0 !important;
            white-space: normal !important;
            word-break: keep-all !important;
            overflow-wrap: anywhere !important;
            text-align: center !important;
        }

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
            border-color: #808080 #ffffff #ffffff #808080 !important;
            background: #ffff00 !important;
            color: #000000 !important;
            text-shadow: none !important;
        }

        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] *,
        div[data-testid="stTabs"] [role="tab"][aria-selected="true"] p {
            color: #000000 !important;
            text-shadow: none !important;
        }

        div[data-testid="stAlert"],
        details[data-testid="stExpander"],
        div[data-testid="stDataFrame"],
        div[data-testid="stTable"],
        div[data-testid="stPlotlyChart"],
        div[data-testid="stPydeckChart"],
        div[data-testid="stVegaLiteChart"] {
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
        }

        div[data-testid="stPlotlyChart"],
        div[data-testid="stPydeckChart"] {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            padding: 1.05rem 1.05rem 2.35rem !important;
            margin-bottom: 1.45rem !important;
            overflow: hidden !important;
        }

        div[data-testid="stPlotlyChart"] > div,
        div[data-testid="stPlotlyChart"] .js-plotly-plot,
        div[data-testid="stPlotlyChart"] .plot-container,
        div[data-testid="stPlotlyChart"] .svg-container {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
        }

        div[data-testid="stHorizontalBlock"] > div:nth-child(even) div[data-testid="stPlotlyChart"],
        div[data-testid="stHorizontalBlock"] > div:nth-child(odd) div[data-testid="stPlotlyChart"] {
            margin-top: 0 !important;
        }

        div[data-testid="stPlotlyChart"] .modebar,
        div[data-testid="stPlotlyChart"] .modebar-container {
            display: none !important;
        }

        .soft-metric-card {
            height: 132px !important;
            min-height: 132px !important;
            padding: 1.75rem 0.75rem 0.82rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
            position: relative !important;
            display: flex !important;
            flex-direction: column !important;
            justify-content: flex-start !important;
            gap: 0.42rem !important;
            overflow: hidden !important;
        }

        .soft-metric-card::before {
            content: "COUNT.EXE" !important;
            display: block !important;
            position: absolute !important;
            left: 0 !important;
            right: 0 !important;
            top: 0 !important;
            height: 1.25rem !important;
            padding-left: 0.35rem !important;
            background: linear-gradient(90deg, #000080, #1084d0) !important;
            color: #ffffff !important;
            font-family: "MS Sans Serif", Tahoma, sans-serif !important;
            font-size: 0.72rem !important;
            font-style: normal !important;
            font-weight: 800 !important;
            line-height: 1.25rem !important;
        }

        .soft-metric-card::after {
            left: 0.45rem !important;
            right: 0.45rem !important;
            bottom: 0.25rem !important;
            height: 3px !important;
            background: repeating-linear-gradient(45deg, #ffff00, #ffff00 8px, #000000 8px, #000000 16px) !important;
            box-shadow: none !important;
            opacity: 1 !important;
        }

        .soft-metric-card__label {
            color: #000000 !important;
            font-family: "MS Sans Serif", Tahoma, sans-serif !important;
            font-size: 0.74rem !important;
            font-weight: 800 !important;
            letter-spacing: 0 !important;
            text-transform: none !important;
            min-height: 1.08rem !important;
            line-height: 1.2 !important;
            overflow: hidden !important;
        }

        .soft-metric-card__value {
            display: inline-block !important;
            max-width: 100% !important;
            width: fit-content !important;
            min-height: 2rem !important;
            color: #00ff00 !important;
            background: #000000 !important;
            border: 2px solid !important;
            border-color: #808080 #ffffff #ffffff #808080 !important;
            padding: 0.2rem 0.5rem !important;
            font-family: "Courier New", Courier, monospace !important;
            font-size: clamp(1.1rem, 1.45vw, 1.55rem) !important;
            font-style: normal !important;
            font-weight: 800 !important;
            line-height: 1.22 !important;
            word-break: keep-all !important;
            overflow-wrap: anywhere !important;
            overflow: visible !important;
            text-shadow: none !important;
        }

        .soft-metric-card__value--compact {
            white-space: nowrap !important;
            overflow-wrap: normal !important;
            word-break: normal !important;
            font-size: clamp(1rem, 1.22vw, 1.26rem) !important;
            letter-spacing: 0 !important;
        }

        .soft-metric-card__value--text-long {
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
            font-size: clamp(0.68rem, 0.82vw, 0.86rem) !important;
            line-height: 1.1 !important;
            max-height: 3.9rem !important;
        }

        .soft-metric-card__value--text-xlong {
            display: block !important;
            width: 100% !important;
            box-sizing: border-box !important;
            font-size: clamp(0.6rem, 0.72vw, 0.76rem) !important;
            line-height: 1.08 !important;
            max-height: 4.15rem !important;
            padding-left: 0.32rem !important;
            padding-right: 0.32rem !important;
        }

        .soft-metric-card__delta {
            background: #ffff00 !important;
            color: #ff0000 !important;
            border: 2px solid #000000 !important;
            box-shadow: none !important;
            min-height: 1.7rem !important;
            padding: 0.24rem 0.36rem !important;
            line-height: 1.2 !important;
            overflow: visible !important;
        }

        .weather-metric-align-spacer {
            height: 1.58rem !important;
            margin: 0 0 0.2rem !important;
        }

        .yoy-table-wrap {
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
        }

        .yoy-table thead th {
            background: #000080 !important;
            color: #ffffff !important;
            border: 1px solid #000000 !important;
        }

        .yoy-table tbody td {
            background: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #808080 !important;
        }

        .pretty-table-window {
            margin: 1rem 0 1.5rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #c0c0c0 !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
        }

        .pretty-table-titlebar {
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.75rem !important;
            min-height: 1.5rem !important;
            padding: 0.22rem 0.45rem !important;
            background: linear-gradient(90deg, #000080, #1084d0) !important;
            color: #ffffff !important;
            font-family: "MS Sans Serif", Tahoma, sans-serif !important;
            font-size: 0.82rem !important;
            font-weight: 800 !important;
            text-shadow: none !important;
        }

        .pretty-table-titlebar span {
            color: #ffffff !important;
            text-shadow: none !important;
        }

        .pretty-table-meta {
            font-family: "Courier New", Courier, monospace !important;
            font-size: 0.7rem !important;
            white-space: nowrap !important;
        }

        .pretty-table-scroll {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto !important;
            overflow-y: visible !important;
            box-sizing: border-box !important;
            padding: 0.65rem !important;
        }

        .pretty-table {
            width: 100% !important;
            min-width: 100% !important;
            table-layout: auto !important;
            border-collapse: collapse !important;
            background: #ffffff !important;
            color: #000000 !important;
            font-family: "MS Sans Serif", Tahoma, "Malgun Gothic", sans-serif !important;
            font-size: 0.86rem !important;
        }

        .pretty-table th {
            padding: 0.62rem 0.78rem !important;
            border: 1px solid #000000 !important;
            background: #ffffcc !important;
            color: #000000 !important;
            font-weight: 900 !important;
            text-align: left !important;
            white-space: normal !important;
            word-break: keep-all !important;
            overflow-wrap: anywhere !important;
        }

        .pretty-table td {
            padding: 0.58rem 0.78rem !important;
            border: 1px solid #808080 !important;
            background: #ffffff !important;
            color: #000000 !important;
            vertical-align: middle !important;
            line-height: 1.28 !important;
            word-break: keep-all !important;
            overflow-wrap: anywhere !important;
        }

        .pretty-table tr:nth-child(even) td {
            background: #e8e8e8 !important;
        }

        .pretty-table .number {
            text-align: right !important;
            font-family: "Courier New", Courier, monospace !important;
            white-space: nowrap !important;
        }

        .pretty-table .name {
            min-width: 12rem !important;
            max-width: 26rem !important;
            font-weight: 800 !important;
            word-break: keep-all !important;
            overflow-wrap: anywhere !important;
        }

        .pretty-table .badge {
            display: inline-block !important;
            padding: 0.16rem 0.42rem !important;
            border: 1px solid #000000 !important;
            background: #ffff00 !important;
            color: #000000 !important;
            font-weight: 800 !important;
            white-space: nowrap !important;
        }

        .pretty-table-note {
            margin: 0 0.45rem 0.45rem !important;
            padding: 0.28rem 0.4rem !important;
            border: 1px solid #808080 !important;
            background: #ffffcc !important;
            color: #000000 !important;
            font-size: 0.78rem !important;
        }

        .map-legend-panel {
            margin: 0.75rem 0 1rem !important;
            padding: 0.65rem 0.75rem !important;
            border: 2px solid !important;
            border-color: #ffffff #808080 #808080 #ffffff !important;
            background: #ffffcc !important;
            box-shadow: inset -1px -1px 0 #404040, inset 1px 1px 0 #dfdfdf !important;
            color: #000000 !important;
            font-family: "MS Sans Serif", Tahoma, "Malgun Gothic", sans-serif !important;
        }

        .map-legend-title {
            margin-bottom: 0.55rem !important;
            font-weight: 900 !important;
        }

        .map-legend-grid {
            display: flex !important;
            flex-wrap: wrap !important;
            gap: 0.55rem !important;
            align-items: stretch !important;
        }

        .map-legend-item {
            display: inline-flex !important;
            align-items: center !important;
            gap: 0.42rem !important;
            min-height: 2rem !important;
            padding: 0.34rem 0.55rem !important;
            border: 1px solid #000000 !important;
            background: #ffffff !important;
            color: #000000 !important;
            font-size: 0.82rem !important;
            font-weight: 800 !important;
            white-space: nowrap !important;
        }

        .map-color-chip {
            display: inline-block !important;
            width: 1rem !important;
            height: 1rem !important;
            border: 2px solid #000000 !important;
            flex: 0 0 auto !important;
        }

        .map-size-dot {
            display: inline-block !important;
            border: 2px solid #000000 !important;
            border-radius: 999px !important;
            background: #ff4048 !important;
            flex: 0 0 auto !important;
        }

        .map-size-dot--small {
            width: 0.45rem !important;
            height: 0.45rem !important;
        }

        .map-size-dot--large {
            width: 0.95rem !important;
            height: 0.95rem !important;
        }

        .map-size-dot--medium {
            width: 0.68rem !important;
            height: 0.68rem !important;
        }

        .map-size-dot--huge {
            width: 1.28rem !important;
            height: 1.28rem !important;
        }

        a, .stMarkdown a {
            color: #0000ff !important;
            text-decoration: underline !important;
        }

        a:hover, .stMarkdown a:hover {
            color: #ff0000 !important;
        }

        .cosmic-footer {
            width: 100% !important;
            max-width: 100% !important;
            box-sizing: border-box !important;
            margin-top: 2rem !important;
            padding: 0.65rem !important;
            border: 4px solid #000000 !important;
            background: repeating-linear-gradient(45deg, #ffff00, #ffff00 10px, #000000 10px, #000000 20px) !important;
            overflow: hidden !important;
        }

        .cosmic-marquee {
            display: flex !important;
            width: max-content !important;
            min-width: max-content !important;
            animation: cosmic-marquee 22s linear infinite !important;
            will-change: transform !important;
            background: #c0c0c0 !important;
            border: 2px solid !important;
            border-color: #808080 #ffffff #ffffff #808080 !important;
            font-family: "Arial Black", Impact, sans-serif !important;
            font-size: clamp(1.15rem, 2.5vw, 2.45rem) !important;
            font-style: normal !important;
            line-height: 1 !important;
            white-space: nowrap !important;
        }

        .cosmic-marquee__track {
            display: flex !important;
            flex: 0 0 auto !important;
            align-items: center !important;
            min-width: max-content !important;
            padding: 0.35rem 0 !important;
        }

        .cosmic-marquee span {
            display: inline-block !important;
            padding-right: 2rem !important;
            color: #0000ff !important;
            background: none !important;
            -webkit-background-clip: initial !important;
            background-clip: initial !important;
            text-shadow: 2px 2px 0 #ffff00 !important;
        }

        .cosmic-footer__meta {
            display: flex !important;
            flex-wrap: wrap !important;
            align-items: center !important;
            justify-content: space-between !important;
            gap: 0.75rem !important;
            color: #000000 !important;
            background: #ffffcc !important;
            border: 2px solid #000000 !important;
            padding: 0.35rem !important;
            font-family: "Courier New", Courier, monospace !important;
        }

        .cosmic-dot {
            background: #00ff00 !important;
            box-shadow: none !important;
        }

        @keyframes cosmic-marquee {
            from { transform: translateX(0); }
            to { transform: translateX(-50%); }
        }

        @keyframes retro-blink {
            0%, 49% { visibility: visible; }
            50%, 100% { visibility: hidden; }
        }

        @media (max-width: 900px) {
            .app-hero__title {
                white-space: normal !important;
            }

            div[data-testid="stTabs"] [data-baseweb="tab-list"] {
                flex-wrap: wrap !important;
            }

            div[data-testid="stTabs"] [role="tab"] {
                flex: 1 1 calc(50% - 0.5rem) !important;
                width: auto !important;
            }
        }

        @media (prefers-reduced-motion: reduce) {
            .section-kicker__eyebrow {
                animation: none !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_app_hero() -> None:
    """앱 첫 화면 제목을 디자인 시스템에 맞는 히어로 영역으로 보여줍니다."""
    st.markdown(
        """
        <section class="app-hero">
            <div class="app-hero__eyebrow">DAEGU BUS DEMAND LAB</div>
            <h1 class="app-hero__title">대구 시내버스 수급 불균형 분석</h1>
            <p class="app-hero__body">
                대구 시내버스 정류소의 시간대별 승하차 데이터를 분석하여 수요가 집중되는 시간과 지역을 확인하고,
                이용 수요 대비 경유 노선 수가 상대적으로 적은 정류소 후보를 탐색합니다.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def inject_heart_easter_egg() -> None:
    """Enter, 이름, Enter 순서가 입력되면 화면에 하트 파티클을 띄웁니다."""
    components.html(
        """
        <script>
        (() => {
            const parentWindow = window.parent;
            const doc = parentWindow.document;
            const secret = "권석현";
            const inputId = "ks-heart-easter-egg-input";
            const layerId = "ks-heart-easter-egg-layer";
            const styleId = "ks-heart-easter-egg-style";

            if (parentWindow.__ksHeartEggCleanup) {
                parentWindow.__ksHeartEggCleanup();
            }

            let timeoutId = null;
            let composing = false;

            function ensureStyle() {
                if (doc.getElementById(styleId)) return;
                const style = doc.createElement("style");
                style.id = styleId;
                style.textContent = `
                    #${layerId} {
                        position: fixed;
                        inset: 0;
                        pointer-events: none;
                        overflow: hidden;
                        z-index: 2147483647;
                    }
                    .ks-heart-easter-egg-heart {
                        position: fixed;
                        left: var(--start-x);
                        bottom: -32px;
                        font-size: var(--size);
                        line-height: 1;
                        color: var(--color);
                        opacity: 0;
                        text-shadow:
                            0 0 10px rgba(255,255,255,.95),
                            0 0 22px rgba(255,56,126,.75);
                        animation: ks-heart-easter-egg-float var(--duration) ease-out forwards;
                        will-change: transform, opacity;
                    }
                    @keyframes ks-heart-easter-egg-float {
                        0% {
                            transform: translate3d(0, 0, 0) scale(.45) rotate(0deg);
                            opacity: 0;
                        }
                        12% {
                            opacity: 1;
                        }
                        100% {
                            transform:
                                translate3d(var(--move-x), var(--move-y), 0)
                                scale(var(--scale))
                                rotate(var(--rotate));
                            opacity: 0;
                        }
                    }
                `;
                doc.head.appendChild(style);
            }

            function ensureLayer() {
                let layer = doc.getElementById(layerId);
                if (!layer) {
                    layer = doc.createElement("div");
                    layer.id = layerId;
                    doc.body.appendChild(layer);
                }
                return layer;
            }

            function ensureInput() {
                let input = doc.getElementById(inputId);
                if (!input) {
                    input = doc.createElement("input");
                    input.id = inputId;
                    input.type = "text";
                    input.autocomplete = "off";
                    input.setAttribute("aria-hidden", "true");
                    input.style.cssText = [
                        "position:fixed",
                        "left:0",
                        "bottom:0",
                        "width:1px",
                        "height:1px",
                        "opacity:0.01",
                        "border:0",
                        "padding:0",
                        "background:transparent",
                        "color:transparent",
                        "caret-color:transparent",
                        "z-index:-1"
                    ].join(";");
                    doc.body.appendChild(input);
                }
                return input;
            }

            function burstHearts() {
                ensureStyle();
                const layer = ensureLayer();
                const hearts = ["❤", "♥", "♡"];
                const colors = ["#ff3b6f", "#ff5ca8", "#ff7a18", "#ffd400", "#89aacc", "#ffffff"];
                const count = 92;

                for (let i = 0; i < count; i += 1) {
                    const heart = doc.createElement("span");
                    heart.className = "ks-heart-easter-egg-heart";
                    heart.textContent = hearts[Math.floor(Math.random() * hearts.length)];
                    heart.style.setProperty("--start-x", `${8 + Math.random() * 84}vw`);
                    heart.style.setProperty("--move-x", `${(Math.random() - 0.5) * 420}px`);
                    heart.style.setProperty("--move-y", `${-180 - Math.random() * 520}px`);
                    heart.style.setProperty("--size", `${18 + Math.random() * 34}px`);
                    heart.style.setProperty("--scale", `${0.8 + Math.random() * 1.35}`);
                    heart.style.setProperty("--rotate", `${(Math.random() - 0.5) * 110}deg`);
                    heart.style.setProperty("--duration", `${1.7 + Math.random() * 1.4}s`);
                    heart.style.setProperty("--color", colors[Math.floor(Math.random() * colors.length)]);
                    heart.style.animationDelay = `${Math.random() * 0.28}s`;
                    layer.appendChild(heart);
                    window.setTimeout(() => heart.remove(), 3600);
                }
            }

            function cancelCapture() {
                const input = doc.getElementById(inputId);
                if (input) {
                    input.value = "";
                    input.blur();
                }
                if (timeoutId) {
                    window.clearTimeout(timeoutId);
                    timeoutId = null;
                }
            }

            function startCapture() {
                const input = ensureInput();
                input.value = "";
                input.focus({ preventScroll: true });
                if (timeoutId) window.clearTimeout(timeoutId);
                timeoutId = window.setTimeout(cancelCapture, 9000);
            }

            function isEditableTarget(target) {
                if (!target || target.id === inputId) return false;
                const tag = (target.tagName || "").toLowerCase();
                if (["input", "textarea", "select"].includes(tag)) return true;
                if (target.isContentEditable) return true;
                return Boolean(target.closest && target.closest("[contenteditable='true'], [role='textbox']"));
            }

            function onDocumentKeydown(event) {
                if (event.key !== "Enter") return;
                if (event.target && event.target.id === inputId) return;
                if (isEditableTarget(event.target)) return;
                event.preventDefault();
                startCapture();
            }

            function onInputKeydown(event) {
                if (event.key === "Escape") {
                    event.preventDefault();
                    cancelCapture();
                    return;
                }
                if (event.key !== "Enter" || event.isComposing || composing) return;
                event.preventDefault();
                const value = event.currentTarget.value.trim().normalize("NFC");
                if (value === secret) {
                    burstHearts();
                }
                cancelCapture();
            }

            function onCompositionStart() {
                composing = true;
            }

            function onCompositionEnd() {
                composing = false;
            }

            ensureStyle();
            const input = ensureInput();
            doc.addEventListener("keydown", onDocumentKeydown, true);
            input.addEventListener("compositionstart", onCompositionStart);
            input.addEventListener("compositionend", onCompositionEnd);
            input.addEventListener("keydown", onInputKeydown);

            parentWindow.__ksHeartEggCleanup = () => {
                doc.removeEventListener("keydown", onDocumentKeydown, true);
                input.removeEventListener("compositionstart", onCompositionStart);
                input.removeEventListener("compositionend", onCompositionEnd);
                input.removeEventListener("keydown", onInputKeydown);
                if (timeoutId) window.clearTimeout(timeoutId);
            };
        })();
        </script>
        """,
        height=0,
    )


def render_marquee_footer() -> None:
    """대시보드 하단에 반복해서 흐르는 마키 푸터를 표시합니다."""
    st.markdown(
        """
        <footer class="cosmic-footer">
            <div class="cosmic-marquee" aria-hidden="true">
                <div class="cosmic-marquee__track">
                    <span>DAEGU BUS DEMAND INTELLIGENCE • ROUTES • STOPS • HOURS • WEATHER •</span>
                    <span>DAEGU BUS DEMAND INTELLIGENCE • ROUTES • STOPS • HOURS • WEATHER •</span>
                </div>
                <div class="cosmic-marquee__track">
                    <span>DAEGU BUS DEMAND INTELLIGENCE • ROUTES • STOPS • HOURS • WEATHER •</span>
                    <span>DAEGU BUS DEMAND INTELLIGENCE • ROUTES • STOPS • HOURS • WEATHER •</span>
                </div>
            </div>
            <div class="cosmic-footer__meta">
                <div><span class="cosmic-dot"></span>현재 필터 기준 분석 결과를 표시 중입니다.</div>
                <div>대구 시내버스 데이터 분석 대시보드</div>
            </div>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(eyebrow: str, title: str, body: str = "") -> None:
    """탭 내부를 탐험 갤러리처럼 구분하는 섹션 헤더를 표시합니다."""
    body_html = f"<div class='section-kicker__body'>{escape(body)}</div>" if body else ""
    st.markdown(
        f"""
        <section class="section-kicker">
            <div class="section-kicker__eyebrow">{escape(eyebrow)}</div>
            <h2 class="section-kicker__title">{escape(title)}</h2>
            {body_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def readable_bar_text_color(marker_color) -> str:
    """노란 막대 안의 숫자는 검정색으로 바꿔 가독성을 확보합니다."""
    if isinstance(marker_color, (list, tuple)) and marker_color:
        marker_color = marker_color[0]
    color = str(marker_color or "").lower()
    if color in {"#ffff00", "yellow", "rgb(255, 255, 0)", "rgba(255, 255, 0, 1)"}:
        return "#000000"
    return "#FFFFFF"


def style_plotly_figure(fig):
    """Plotly 그래프가 프리미엄 다크 대시보드 배경과 자연스럽게 섞이도록 공통 스타일을 적용합니다."""
    if fig is None:
        return None
    for trace in fig.data:
        if getattr(trace, "type", None) == "bar" and getattr(trace, "orientation", None) == "h":
            trace.update(
                textposition="inside",
                insidetextanchor="end",
                textfont=dict(
                    color=readable_bar_text_color(getattr(trace.marker, "color", None)),
                    size=12,
                    family="Courier New, monospace",
                ),
                cliponaxis=False,
            )
            fig.update_yaxes(categoryorder="total ascending")
        if getattr(trace, "type", None) == "bar":
            trace.update(
                marker_line_width=2,
                marker_line_color="#000000",
                opacity=1,
                textfont=dict(color=readable_bar_text_color(getattr(trace.marker, "color", None))),
            )
        if getattr(trace, "type", None) == "scatter":
            trace.update(line=dict(width=3), marker=dict(size=7, line=dict(width=2, color="#000000")))
    current_height = fig.layout.height
    if current_height is None or current_height < 540:
        fig.update_layout(height=540)
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#C0C0C0",
        plot_bgcolor="#FFFFFF",
        autosize=True,
        font=dict(family="MS Sans Serif, Tahoma, Malgun Gothic, sans-serif", color="#000000", size=13),
        title=dict(
            font=dict(family="Arial Black, Impact, Malgun Gothic, sans-serif", color="#000000", size=20),
            x=0.02,
            xanchor="left",
            pad=dict(t=10, b=20),
        ),
        legend=dict(
            bgcolor="#FFFFCC",
            bordercolor="#000000",
            borderwidth=1,
            font=dict(color="#000000"),
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
        ),
        colorway=["#0000FF", "#00AA00", "#FF0000", "#FFFF00", "#800080", "#008080", "#808080"],
        hoverlabel=dict(
            bgcolor="#FFFFCC",
            bordercolor="#000000",
            font=dict(color="#000000", family="MS Sans Serif, Tahoma, Malgun Gothic, sans-serif"),
        ),
        margin=dict(l=82, r=108, t=90, b=210),
    )
    fig.update_xaxes(
        color="#000000",
        gridcolor="#808080",
        zerolinecolor="#000000",
        linecolor="#000000",
        title_font=dict(color="#000000"),
        tickfont=dict(color="#000000"),
        automargin=True,
        title_standoff=26,
    )
    fig.update_yaxes(
        color="#000000",
        gridcolor="#808080",
        zerolinecolor="#000000",
        linecolor="#000000",
        title_font=dict(color="#000000"),
        tickfont=dict(color="#000000"),
        automargin=True,
        title_standoff=30,
    )
    return fig


def build_dark_line_chart(data: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str):
    """Streamlit 기본 라인 차트 대신 공통 Plotly 테마를 적용할 라인 차트를 만듭니다."""
    try:
        import plotly.express as px
    except ImportError:
        return None
    if data.empty or x_col not in data.columns or y_col not in data.columns:
        return None
    fig = px.line(
        data,
        x=x_col,
        y=y_col,
        markers=True,
        title=title,
        labels={x_col: "기간", y_col: y_label},
    )
    fig.update_layout(height=420)
    return fig


def build_yoy_comparison_line(trend: pd.DataFrame):
    """전년 대비 월별 이용량 비교를 Plotly 라인 차트로 만듭니다."""
    try:
        import plotly.express as px
    except ImportError:
        return None
    if trend.empty:
        return None
    chart = trend[["period", "base_value", "target_value"]].rename(
        columns={
            "period": "월",
            "base_value": f"{YOY_BASE_YEAR}년",
            "target_value": f"{YOY_TARGET_YEAR}년",
        }
    )
    melted = chart.melt(id_vars="월", var_name="연도", value_name="승차 인원")
    fig = px.line(
        melted,
        x="월",
        y="승차 인원",
        color="연도",
        markers=True,
        title=f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 승차 인원",
    )
    fig.update_layout(height=430)
    return fig


def build_yoy_growth_bar(trend: pd.DataFrame):
    """전년 동월 대비 증감률을 Plotly 막대 차트로 만듭니다."""
    try:
        import plotly.express as px
    except ImportError:
        return None
    if trend.empty or "growth_rate" not in trend.columns:
        return None
    chart = trend[["period", "growth_rate"]].rename(columns={"period": "월", "growth_rate": "전년 동월 대비 증감률"})
    fig = px.bar(
        chart,
        x="월",
        y="전년 동월 대비 증감률",
        title="전년 동월 대비 증감률",
        text="전년 동월 대비 증감률",
    )
    fig.update_traces(texttemplate="%{text:+.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(height=390)
    return fig


def render_soft_metric(label: str, value, delta: str | None = None) -> None:
    """긴 정류소명도 잘리지 않는 커스텀 지표 카드를 표시합니다."""
    value_text = "-" if value is None else str(value)
    compact_value = any(char.isdigit() for char in value_text) and len(value_text) >= 4
    value_classes = ["soft-metric-card__value"]
    if compact_value:
        value_classes.append("soft-metric-card__value--compact")
    elif len(value_text) >= 11:
        value_classes.append("soft-metric-card__value--text-xlong")
    elif len(value_text) >= 7:
        value_classes.append("soft-metric-card__value--text-long")
    value_class = " ".join(value_classes)
    delta_html = f"<div class='soft-metric-card__delta'>{escape(str(delta))}</div>" if delta else ""
    st.markdown(
        f"""
        <div class="soft-metric-card">
            <div class="soft-metric-card__label">{escape(str(label))}</div>
            <div class="{value_class}">{escape(value_text)}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


TABLE_COLUMN_LABELS = {
    "_merge_key": "병합 기준",
    "stop_id": "정류소 ID",
    "stop_name": "정류소명",
    "stop_label": "정류소명",
    "district": "구·군",
    "boardings": "승차 인원",
    "alightings": "하차 인원",
    "total_riders": "전체 이용객",
    "route_count": "경유 노선 수",
    "boardings_per_route": "노선당 승차 인원",
    "morning_boardings": "출근 시간 승차",
    "evening_boardings": "퇴근 시간 승차",
    "morning_concentration": "출근 집중도",
    "evening_concentration": "퇴근 집중도",
    "peak_hour": "최대 혼잡 시간",
    "peak_hour_label": "최대 혼잡 시간대",
    "stop_type": "정류소 유형",
    "year": "연도",
    "month": "월",
    "period": "연월",
    "base_value": f"{YOY_BASE_YEAR}년 승차 인원",
    "target_value": f"{YOY_TARGET_YEAR}년 승차 인원",
    "change": "증감 인원",
    "growth_rate": "증감률",
    "previous_year_value": "전년 동월",
    "yoy_growth_rate": "전년 대비 증감률",
    "passengers": "이용객",
    "cluster": "군집",
    "cluster_name": "군집 유형",
}


PERCENT_COLUMNS = {
    "morning_concentration",
    "evening_concentration",
    "growth_rate",
    "yoy_growth_rate",
}


NUMERIC_TABLE_COLUMNS = {
    "boardings",
    "alightings",
    "total_riders",
    "route_count",
    "boardings_per_route",
    "morning_boardings",
    "evening_boardings",
    "base_value",
    "target_value",
    "change",
    "previous_year_value",
    "passengers",
}


def table_column_label(column) -> str:
    """내부 컬럼명을 발표용 한글 컬럼명으로 바꿉니다."""
    text = str(column)
    return TABLE_COLUMN_LABELS.get(text, text)


def clean_merge_key_value(value) -> str:
    """표에 보이는 병합 기준에서 내부 접두어를 제거합니다."""
    text = str(value)
    if text.startswith("id:"):
        return text.replace("id:", "", 1)
    if text.startswith("name:"):
        cleaned = text.replace("name:", "", 1).strip("|")
        parts = [part for part in cleaned.split("|") if part]
        return " / ".join(parts) if parts else cleaned
    return text


def table_value_is_number(value) -> bool:
    """표 셀 값이 숫자로 표시 가능한지 확인합니다."""
    if value is None or pd.isna(value):
        return False
    try:
        float(str(value).replace(",", "").replace("%", ""))
        return True
    except (TypeError, ValueError):
        return False


def format_pretty_table_value(value, column) -> str:
    """표 안의 숫자와 비율을 사람이 읽기 좋은 단위로 바꿉니다."""
    if value is None or pd.isna(value):
        return "-"

    column_key = str(column)
    if column_key == "_merge_key":
        return clean_merge_key_value(value)

    if isinstance(value, str) and "%" in value:
        return value

    if column_key in {"year", "month", "cluster"}:
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)

    if column_key in PERCENT_COLUMNS:
        try:
            return f"{float(value):,.1f}%"
        except (TypeError, ValueError):
            return str(value)

    if column_key in NUMERIC_TABLE_COLUMNS or table_value_is_number(value):
        text_value = str(value).replace(",", "").replace("%", "")
        try:
            number = float(text_value)
        except (TypeError, ValueError):
            return str(value)
        if abs(number - round(number)) < 0.001:
            return format_number(number)
        return format_number(number, decimals=1)

    return str(value)


def render_pretty_table(table: pd.DataFrame, title: str, max_rows: int = 20) -> None:
    """Streamlit 기본 데이터프레임 대신 발표 화면에 어울리는 HTML 표를 표시합니다."""
    if table is None or table.empty:
        empty_message(f"{title} 데이터가 없습니다.")
        return

    display = table.head(max_rows).copy()
    headers = "".join(f"<th>{escape(table_column_label(col))}</th>" for col in display.columns)
    rows = []
    for _, row in display.iterrows():
        cells = []
        for col in display.columns:
            raw_value = row[col]
            text_value = format_pretty_table_value(raw_value, col)
            cell_classes = []
            if str(col) in {"stop_name", "stop_label"}:
                cell_classes.append("name")
            if str(col) in {"stop_type", "cluster_name"}:
                text_value = f"<span class='badge'>{escape(text_value)}</span>"
            else:
                text_value = escape(text_value)
            if str(col) in NUMERIC_TABLE_COLUMNS or str(col) in PERCENT_COLUMNS or table_value_is_number(raw_value):
                cell_classes.append("number")
            class_attr = f" class='{' '.join(cell_classes)}'" if cell_classes else ""
            cells.append(f"<td{class_attr}>{text_value}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")

    note = ""
    if len(table) > len(display):
        note = (
            f"<div class='pretty-table-note'>상위 {len(display):,}개 행만 표시합니다. "
            f"전체 {len(table):,}개 행은 CSV 다운로드로 확인할 수 있습니다.</div>"
        )

    st.markdown(
        f"""
        <div class="pretty-table-window">
            <div class="pretty-table-titlebar">
                <span>{escape(str(title))}</span>
                <span class="pretty-table-meta">{len(table):,} ROWS</span>
            </div>
            <div class="pretty-table-scroll">
                <table class="pretty-table">
                    <thead><tr>{headers}</tr></thead>
                    <tbody>{''.join(rows)}</tbody>
                </table>
            </div>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_tables_grid(tables: dict, max_rows: int = 12) -> None:
    """핵심 분석 표를 너무 큰 빈 영역 없이 1열 또는 2열로 배치합니다."""
    narrow_buffer: list[tuple[str, pd.DataFrame]] = []

    def flush_narrow_buffer() -> None:
        nonlocal narrow_buffer
        while narrow_buffer:
            pair = narrow_buffer[:2]
            narrow_buffer = narrow_buffer[2:]
            cols = st.columns(len(pair))
            for col, (title, table) in zip(cols, pair):
                with col:
                    render_pretty_table(table, title, max_rows=max_rows)

    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        if len(table.columns) <= 3:
            narrow_buffer.append((str(name), table))
            if len(narrow_buffer) == 2:
                flush_narrow_buffer()
        else:
            flush_narrow_buffer()
            render_pretty_table(table, str(name), max_rows=max_rows)
    flush_narrow_buffer()


def build_core_analysis_tables(
    stop_df: pd.DataFrame,
    hourly_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    top_n: int,
) -> dict:
    """핵심 분석 표를 만들고 장기 증감 표는 2025년 대비 2026년 비교만 남깁니다."""
    tables = create_analysis_tables(stop_df, hourly_df, monthly_df, top_n=top_n)
    tables = {
        name: table
        for name, table in tables.items()
        if "이용량 증가율" not in str(name) and "이용량 감소율" not in str(name)
    }

    metric = "boardings" if "boardings" in monthly_df.columns else "passengers"
    target_trend = build_target_year_yoy_trend(monthly_df, metric)
    if not target_trend.empty:
        tables[f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 증감 요약"] = target_trend
    return tables


def render_stop_type_criteria(stop_df: pd.DataFrame) -> None:
    """정류소 유형이 어떤 기준으로 분류되는지 앱 화면에 설명합니다."""
    low_cutoff_text = "계산 불가"
    if not stop_df.empty and "boardings" in stop_df.columns and stop_df["boardings"].notna().any():
        low_cutoff = stop_df["boardings"].quantile(LOW_USAGE_QUANTILE)
        low_cutoff_text = f"{format_number(low_cutoff)}명 이하"

    criteria = pd.DataFrame(
        [
            {
                "정류소 유형": "저이용형",
                "분류 기준": f"전체 승차 인원이 현재 필터 데이터의 하위 {LOW_USAGE_QUANTILE:.0%} 이하",
                "현재 기준값": low_cutoff_text,
                "해석": "이용량 자체가 적은 정류소입니다. 다른 시간대 집중 조건보다 먼저 적용됩니다.",
            },
            {
                "정류소 유형": "출퇴근형",
                "분류 기준": f"출근 집중도 {HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상 + 퇴근 집중도 {HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상",
                "현재 기준값": f"각각 {HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상",
                "해석": "오전과 저녁 피크가 모두 뚜렷한 정류소입니다.",
            },
            {
                "정류소 유형": "출근형",
                "분류 기준": f"출근 집중도 {HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상 + 출근 집중도가 퇴근 집중도보다 큼",
                "현재 기준값": f"{HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상",
                "해석": "오전 7~9시 승차 비중이 상대적으로 큰 정류소입니다.",
            },
            {
                "정류소 유형": "퇴근형",
                "분류 기준": f"퇴근 집중도 {HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상 + 퇴근 집중도가 출근 집중도보다 큼",
                "현재 기준값": f"{HIGH_CONCENTRATION_THRESHOLD:.0f}% 이상",
                "해석": "오후 5~8시 승차 비중이 상대적으로 큰 정류소입니다.",
            },
            {
                "정류소 유형": "생활형",
                "분류 기준": "저이용형, 출퇴근형, 출근형, 퇴근형 조건에 해당하지 않음",
                "현재 기준값": "-",
                "해석": "특정 피크 시간대에 과도하게 몰리지 않는 정류소입니다.",
            },
            {
                "정류소 유형": "분류 불가",
                "분류 기준": "승차 인원 또는 출근·퇴근 집중도 계산에 필요한 값이 부족함",
                "현재 기준값": "-",
                "해석": "현재 데이터만으로 시간대 유형을 안정적으로 판단하기 어렵습니다.",
            },
        ]
    )

    with st.expander("정류소 유형 분류 기준 보기"):
        st.caption(
            f"출근 시간은 {MORNING_START_HOUR}시부터 {MORNING_END_HOUR}시까지, "
            f"퇴근 시간은 {EVENING_START_HOUR}시부터 {EVENING_END_HOUR}시까지로 계산합니다. "
            "집중도는 해당 시간대 승차 인원 / 전체 승차 인원 × 100입니다."
        )
        render_pretty_table(criteria, "정류소 유형 분류 기준", max_rows=10)


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


def processed_fingerprint(processed_dir: Path) -> tuple:
    """정제 CSV가 바뀌면 Streamlit 캐시가 갱신되도록 파일 상태를 요약합니다."""
    if not processed_dir.exists():
        return tuple()
    fingerprint = []
    for file_name in sorted(set(PROCESSED_CACHE_FILES.values())):
        path = processed_dir / file_name
        if path.exists():
            stat = path.stat()
            fingerprint.append((file_name, int(stat.st_mtime), stat.st_size))
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
def load_project_data(data_dir: str, raw_fingerprint: tuple, cache_fingerprint: tuple) -> dict:
    """Streamlit에서는 원본 CSV가 아니라 정제된 CSV만 불러옵니다."""
    data_path = Path(data_dir)
    del raw_fingerprint, cache_fingerprint
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


def get_project_data() -> dict:
    """한 번 불러온 정제 데이터를 세션에 저장해 반복 리로드를 줄입니다."""
    current_fingerprint = (data_fingerprint(DATA_DIR), processed_fingerprint(PROCESSED_DIR))
    if (
        "project_data_bundle" in st.session_state
        and st.session_state.get("project_data_fingerprint") == current_fingerprint
    ):
        return st.session_state["project_data_bundle"]

    bundle = load_project_data(str(DATA_DIR), current_fingerprint[0], current_fingerprint[1])
    st.session_state["project_data_bundle"] = bundle
    st.session_state["project_data_fingerprint"] = current_fingerprint
    return bundle


def reload_project_data() -> dict:
    """사용자가 원할 때만 정제 CSV를 다시 읽습니다."""
    st.session_state.pop("project_data_bundle", None)
    st.session_state.pop("project_data_fingerprint", None)
    load_project_data.clear()
    return get_project_data()


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
    if hourly_df.empty or stop_df.empty:
        return pd.DataFrame()
    if "_merge_key" in hourly_df.columns and "_merge_key" in stop_df.columns:
        valid_keys = set(stop_df["_merge_key"].dropna())
        filtered = hourly_df[hourly_df["_merge_key"].isin(valid_keys)].copy()
    else:
        filtered = pd.DataFrame()
    if filtered.empty and "stop_name" in hourly_df.columns and "stop_name" in stop_df.columns:
        valid_names = set(stop_df["stop_name"].dropna().astype(str))
        filtered = hourly_df[hourly_df["stop_name"].astype(str).isin(valid_names)].copy()
    metadata_cols = [
        col
        for col in ["district", "stop_type", "route_count"]
        if col in stop_df.columns and col not in filtered.columns
    ]
    if metadata_cols and not filtered.empty:
        if "_merge_key" in filtered.columns and "_merge_key" in stop_df.columns:
            metadata = stop_df[["_merge_key"] + metadata_cols].dropna(subset=["_merge_key"]).drop_duplicates("_merge_key")
            filtered = filtered.merge(metadata, on="_merge_key", how="left")
        elif "stop_name" in filtered.columns and "stop_name" in stop_df.columns:
            metadata = stop_df[["stop_name"] + metadata_cols].dropna(subset=["stop_name"]).drop_duplicates("stop_name")
            filtered = filtered.merge(metadata, on="stop_name", how="left")
    if not filtered.empty and "stop_name" in filtered.columns and "stop_name" in stop_df.columns:
        refill_cols = [
            col
            for col in ["district", "stop_type", "route_count"]
            if col in stop_df.columns and (col not in filtered.columns or filtered[col].notna().sum() == 0)
        ]
        if refill_cols:
            filtered = filtered.drop(columns=[col for col in refill_cols if col in filtered.columns], errors="ignore")
            metadata = stop_df[["stop_name"] + refill_cols].dropna(subset=["stop_name"]).drop_duplicates("stop_name")
            filtered = filtered.merge(metadata, on="stop_name", how="left")
    if "total_riders" not in filtered.columns:
        if {"boardings", "alightings"}.issubset(filtered.columns):
            filtered["total_riders"] = filtered["boardings"].fillna(0) + filtered["alightings"].fillna(0)
        elif "passengers" in filtered.columns:
            filtered["total_riders"] = filtered["passengers"]
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


def show_metric(label: str, value, decimals: int = 0) -> None:
    """숫자 지표를 천 단위 쉼표와 함께 표시합니다."""
    render_soft_metric(label, format_number(value, decimals=decimals))


def show_text_metric(label: str, value, delta: str | None = None) -> None:
    """정류소명처럼 긴 텍스트 지표를 줄바꿈 가능한 카드로 표시합니다."""
    render_soft_metric(label, "-" if value is None else value, delta=delta)


def render_plotly_chart(fig, empty_text: str, caption: bool = False) -> None:
    """그래프 객체가 있을 때만 표시하고 Streamlit 내부 객체가 화면에 노출되지 않게 합니다."""
    if fig is not None:
        style_plotly_figure(fig)
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": False, "responsive": True},
        )
    elif caption:
        st.caption(empty_text)
    else:
        empty_message(empty_text)


def render_two_column_charts(chart_items: list[tuple]) -> None:
    """Plotly 그래프를 한 행에 두 개씩 배치해 가로 공간을 꽉 채웁니다."""
    for start_index in range(0, len(chart_items), 2):
        row_items = chart_items[start_index:start_index + 2]
        columns = st.columns(2, gap="large")

        for column_index, column in enumerate(columns):
            if column_index >= len(row_items):
                continue

            fig, empty_text, *options = row_items[column_index]
            caption = bool(options[0]) if options else False

            with column:
                render_plotly_chart(fig, empty_text, caption=caption)


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

    render_section_header(
        "Overview",
        "전체 현황",
        "승차·하차 수요는 2024~2025년 월별 이용자수 데이터 기준이며, 시간대 피크와 집중도는 시간대별 승하차 데이터로 함께 비교합니다.",
    )

    metric_cols = st.columns(6, gap="small")
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
        show_text_metric("가장 이용객이 많은 정류소", top_stop)
    with metric_cols[4]:
        peak_hour = "-"
        if not hourly_df.empty and "hour" in hourly_df.columns and "boardings" in hourly_df.columns:
            hourly_total = hourly_df.groupby("hour", as_index=False)["boardings"].sum()
            if not hourly_total.empty:
                peak_hour = f"{int(hourly_total.sort_values('boardings', ascending=False).iloc[0]['hour'])}시"
        show_text_metric("가장 혼잡한 시간대", peak_hour)
    with metric_cols[5]:
        per_route_stop = "-"
        if "boardings_per_route" in stop_df.columns and stop_df["boardings_per_route"].notna().any():
            per_route_stop = stop_df.sort_values("boardings_per_route", ascending=False).iloc[0].get("stop_name", "-")
        show_text_metric("노선당 승차 인원 최고", per_route_stop)

    render_two_column_charts(
        [
            (
                plot_top_stops(
                    stop_df,
                    metric=metric_col,
                    n=top_n,
                    title=f"정류소별 {plot_metric_label(metric_col)} TOP {top_n}",
                ),
                "정류소 순위 그래프를 그릴 수 없습니다.",
            ),
            (
                plot_district_bar(stop_df, metric=metric_col, title="구·군별 승차 인원"),
                "구·군 컬럼이 없어 구·군별 그래프를 표시할 수 없습니다.",
            ),
            (
                plot_hourly_line(hourly_df, metric=metric_col, title="시간대별 승차 인원"),
                "시간대 컬럼이 없어 시간대별 그래프를 표시할 수 없습니다.",
            ),
            (
                plot_type_pie(stop_df),
                "정류소 유형을 계산할 수 없습니다.",
            ),
        ]
    )
    render_stop_type_criteria(stop_df)


def hourly_tab(stop_df: pd.DataFrame, hourly_df: pd.DataFrame, top_n: int, metric_col: str) -> None:
    """시간대별 분석 탭을 구성합니다."""
    if hourly_df.empty:
        empty_message("시간대 컬럼이 있는 데이터가 없어 시간대별 분석을 표시할 수 없습니다.")
        return

    render_section_header(
        "Time Flow",
        "시간대별 이용 패턴",
        "하루의 이동 리듬을 시간대, 정류소, 구·군 단위로 쪼개어 피크와 반복 패턴을 확인합니다.",
    )

    morning = (
        stop_df.sort_values("morning_boardings", ascending=False).head(top_n)
        if "morning_boardings" in stop_df.columns
        else pd.DataFrame()
    )
    evening = (
        stop_df.sort_values("evening_boardings", ascending=False).head(top_n)
        if "evening_boardings" in stop_df.columns
        else pd.DataFrame()
    )

    render_two_column_charts(
        [
            (
                plot_hourly_line(hourly_df, metric=metric_col, title="시간대별 전체 승차 인원"),
                "시간대별 전체 승차 인원을 표시할 수 없습니다.",
            ),
            (
                plot_board_alight_line(hourly_df),
                "승차·하차 비교에 필요한 컬럼이 부족합니다.",
            ),
            (
                plot_stop_hour_heatmap(hourly_df, stop_df, metric=metric_col, top_n=20),
                "정류소 × 시간대 히트맵을 표시할 수 없습니다.",
            ),
            (
                plot_district_hour_heatmap(hourly_df, metric=metric_col),
                "구·군 × 시간대 히트맵을 표시할 수 없습니다.",
            ),
            (
                plot_top_stops(
                    morning,
                    metric="morning_boardings",
                    n=top_n,
                    title=f"출근 시간 이용객 TOP {top_n}",
                ),
                "출근 시간 이용객을 계산할 수 없습니다.",
            ),
            (
                plot_top_stops(
                    evening,
                    metric="evening_boardings",
                    n=top_n,
                    title=f"퇴근 시간 이용객 TOP {top_n}",
                ),
                "퇴근 시간 이용객을 계산할 수 없습니다.",
            ),
        ]
    )

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

    render_section_header(
        "Stop Lens",
        "정류소별 상세 분석",
        "개별 정류소의 수요, 노선 공급, 혼잡 시간대, 월별 흐름을 한 정류소 단위로 좁혀 봅니다.",
    )

    stop_name = st.selectbox("상세 분석 정류소 선택", sorted(stop_df["stop_name"].dropna().astype(str).unique()))
    selected = stop_df[stop_df["stop_name"].astype(str) == stop_name]
    if selected.empty:
        empty_message("선택한 정류소 데이터가 없습니다.")
        return

    row = selected.iloc[0]
    cols = st.columns(5)
    with cols[0]:
        show_text_metric("정류소명", row.get("stop_name", "-"))
    with cols[1]:
        show_text_metric("행정구역", row.get("district", "-"))
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
        show_text_metric("최대 혼잡 시간대", row.get("peak_hour_label", "-"))
    with cols2[2]:
        show_metric("출근 시간 집중도(%)", row.get("morning_concentration"))
    with cols2[3]:
        show_metric("퇴근 시간 집중도(%)", row.get("evening_concentration"))
    with cols2[4]:
        show_text_metric("정류소 유형", row.get("stop_type", "-"))

    fig = plot_stop_comparison_line(hourly_df, [stop_name], metric="boardings", include_average=True)
    render_plotly_chart(fig, "선택 정류소의 시간대별 그래프를 표시할 수 없습니다.")

    if not monthly_df.empty and "_merge_key" in monthly_df.columns:
        selected_keys = selected["_merge_key"].dropna().unique()
        stop_monthly = monthly_df[monthly_df["_merge_key"].isin(selected_keys)].copy()
        if not stop_monthly.empty and {"year", "month"}.issubset(stop_monthly.columns):
            stop_monthly["period"] = stop_monthly["year"].astype(int).astype(str) + "-" + stop_monthly["month"].astype(int).astype(str).str.zfill(2)
            metric = "boardings" if "boardings" in stop_monthly.columns else "passengers"
            monthly_fig = build_dark_line_chart(
                stop_monthly,
                "period",
                metric,
                "선택 정류소 월별 이용량",
                plot_metric_label(metric),
            )
            render_plotly_chart(monthly_fig, "선택 정류소의 월별 그래프를 표시할 수 없습니다.")


def imbalance_tab(stop_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """수요·공급 불균형 분석 탭을 구성하고 후보 정류소를 반환합니다."""
    if stop_df.empty:
        empty_message("수요·공급 불균형 분석에 사용할 데이터가 없습니다.")
        return pd.DataFrame()

    render_section_header(
        "Supply Gap",
        "수요·공급 불균형 탐색",
        "승차 수요는 높지만 경유 노선 수나 노선당 승차 밀도가 비정상적으로 나타나는 후보 정류소를 찾습니다.",
    )

    st.caption("이 분석은 실제 노선 부족을 확정하지 않고, 추가 검토가 필요한 후보 정류소를 찾는 용도입니다.")
    criteria_mode = st.radio(
        "후보 기준 방식",
        ["발표용 절대 기준", "데이터 분위수 기준"],
        horizontal=True,
        help="발표용 절대 기준은 설명이 쉬운 숫자를 직접 사용하고, 데이터 분위수 기준은 전체 정류소 분포의 상위·하위 비율을 사용합니다.",
    )

    if criteria_mode == "발표용 절대 기준":
        col1, col2, col3 = st.columns(3)
        with col1:
            demand_threshold = st.number_input("승차 인원 기준", min_value=0, value=200000, step=10000)
        with col2:
            route_threshold = st.number_input("경유 노선 수 기준", min_value=1, value=3, step=1)
        with col3:
            per_route_threshold = st.number_input("노선당 승차 인원 기준", min_value=0, value=150000, step=10000)

        current_analysis = importlib.reload(analysis_module)
        candidates, thresholds = current_analysis.imbalance_candidates(
            stop_df,
            demand_threshold=demand_threshold,
            route_threshold=route_threshold,
            per_route_threshold=per_route_threshold,
        )
        st.write(
            f"후보 기준: 승차 인원 {format_number(demand_threshold)}명 이상, "
            f"경유 노선 수 {format_number(route_threshold)}개 이하, "
            f"노선당 승차 인원 {format_number(per_route_threshold)}명/노선 이상"
        )
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            demand_q = st.slider("전체 승차 인원 상위 기준", 0.50, 0.95, 0.75, 0.05)
        with col2:
            route_q = st.slider("경유 노선 수 하위 기준", 0.10, 0.90, 0.50, 0.05)
        with col3:
            per_route_q = st.slider("노선당 승차 인원 상위 기준", 0.50, 0.99, 0.90, 0.01)

        current_analysis = importlib.reload(analysis_module)
        candidates, thresholds = current_analysis.imbalance_candidates(
            stop_df,
            demand_quantile=demand_q,
            route_quantile=route_q,
            per_route_quantile=per_route_q,
        )
        st.write(
            f"후보 기준: 승차 인원 상위 {(1 - demand_q) * 100:.0f}% "
            f"({format_number(thresholds.get('demand_threshold'))}명 이상), "
            f"경유 노선 수 하위 {route_q * 100:.0f}% "
            f"({format_number(thresholds.get('route_threshold'))}개 이하), "
            f"노선당 승차 인원 상위 {(1 - per_route_q) * 100:.0f}% "
            f"({format_number(thresholds.get('per_route_threshold'))}명/노선 이상)"
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
    render_pretty_table(candidates[display_cols], "수요·공급 불균형 추가 검토 후보", max_rows=20)
    st.download_button(
        "추가 검토 후보 정류소 CSV 다운로드",
        data=make_download_csv(candidates[display_cols] if display_cols else candidates),
        file_name="imbalance_candidates.csv",
        mime="text/csv",
    )
    return candidates


def map_tab(stop_df: pd.DataFrame, show_map: bool) -> None:
    """지도 분석 탭을 구성합니다."""
    render_section_header(
        "Urban Field",
        "지도 기반 정류소 분포",
        "아이콘 크기는 승차 인원, 아이콘 색상은 노선당 승차 인원 밀도를 반영해 정류소 분포를 공간적으로 탐색합니다.",
    )
    if not show_map:
        empty_message("사이드바에서 지도 표시 여부가 꺼져 있습니다.")
        return
    map_data_count = len(stop_df.dropna(subset=["lat", "lon"])) if {"lat", "lon"}.issubset(stop_df.columns) else 0
    max_icons = None
    if map_data_count > 900:
        max_icons = st.slider(
            "지도 표시 정류소 수",
            min_value=300,
            max_value=int(map_data_count),
            value=min(900, int(map_data_count)),
            step=100,
            help="줌아웃 상태에서 아이콘이 겹치지 않도록 기본값은 이용량 상위 정류소 위주로 표시합니다.",
        )
        st.caption(
            f"현재 지도에는 승차 인원 기준 상위 {format_number(max_icons)}개 정류소를 표시합니다. "
            "표시 개수를 조절하면 상위 이용 정류소부터 단계적으로 후보 분포를 확인할 수 있습니다."
        )
    visible_map_data = pd.DataFrame()
    if {"lat", "lon", "boardings"}.issubset(stop_df.columns):
        visible_map_data = stop_df.dropna(subset=["lat", "lon"]).copy()
        if max_icons is not None and max_icons > 0 and len(visible_map_data) > max_icons:
            visible_map_data = visible_map_data.sort_values("boardings", ascending=False).head(max_icons).copy()

    if not visible_map_data.empty:
        per_route = pd.to_numeric(
            visible_map_data.get("boardings_per_route", pd.Series(0, index=visible_map_data.index)),
            errors="coerce",
        ).fillna(0)
        boardings = pd.to_numeric(visible_map_data["boardings"], errors="coerce").fillna(0).clip(lower=0)
        high_cutoff = per_route.quantile(0.70)
        very_high_cutoff = per_route.quantile(0.90)
        medium_size_cutoff = boardings.quantile(0.50)
        large_size_cutoff = boardings.quantile(0.80)
        huge_size_cutoff = boardings.quantile(0.95)
        if pd.isna(high_cutoff):
            high_cutoff = 0
        if pd.isna(very_high_cutoff) or very_high_cutoff <= high_cutoff:
            very_high_cutoff = per_route.max() if per_route.max() > 0 else high_cutoff + 1
        if pd.isna(medium_size_cutoff):
            medium_size_cutoff = 0
        if pd.isna(large_size_cutoff) or large_size_cutoff <= medium_size_cutoff:
            large_size_cutoff = medium_size_cutoff + 1
        if pd.isna(huge_size_cutoff) or huge_size_cutoff <= large_size_cutoff:
            huge_size_cutoff = large_size_cutoff + 1

    import importlib
    import src.visualization as visualization_module

    visualization_module = importlib.reload(visualization_module)
    deck = visualization_module.create_pydeck_map(stop_df, max_icons=max_icons)
    if deck is None:
        empty_message("위치정보 데이터가 없어 지도 분석을 표시할 수 없습니다.")
        return
    map_key = f"bus_stop_icon_map_v6_{max_icons or 'all'}"
    st.pydeck_chart(deck, use_container_width=True, key=map_key)


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

    st.subheader(f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 승차 인원")
    render_two_column_charts(
        [
            (
                build_yoy_comparison_line(trend),
                f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 비교 차트를 표시할 수 없습니다.",
            ),
            (
                build_yoy_growth_bar(trend),
                "전년 동월 대비 증감률 차트를 표시할 수 없습니다.",
            ),
        ]
    )

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
    render_pretty_table(formatted, f"{YOY_BASE_YEAR}년 대비 {YOY_TARGET_YEAR}년 월별 증감 요약", max_rows=12)


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
        border: 1px solid {SOFT_STROKE};
        border-radius: 26px;
        overflow: hidden;
        background: rgba(20, 20, 22, 0.82);
        box-shadow: {SOFT_SHADOW};
    }}
    .yoy-table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.94rem;
    }}
    .yoy-table thead th {{
        padding: 0.92rem 0.9rem;
        color: {SOFT_TEXT};
        background: rgba(255, 255, 255, 0.045);
        border-bottom: 1px solid rgba(255, 255, 255, 0.10);
        text-align: left;
        white-space: nowrap;
        font-weight: 800;
    }}
    .yoy-table tbody td {{
        padding: 0.86rem 0.9rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.075);
        color: {SOFT_TEXT};
        vertical-align: middle;
    }}
    .yoy-table tbody tr:nth-child(even) {{
        background: rgba(255, 255, 255, 0.025);
    }}
    .yoy-table tbody tr:hover {{
        background: rgba(137, 170, 204, 0.08);
    }}
    .yoy-table .rank {{
        width: 3.3rem;
        color: {SOFT_ACCENT};
        font-weight: 800;
        text-align: center;
    }}
    .yoy-table .name {{
        color: {SOFT_TEXT};
        font-weight: 800;
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
        padding: 0.28rem 0.68rem;
        border-radius: 999px;
        font-weight: 800;
        font-variant-numeric: tabular-nums;
        box-shadow: {SOFT_SHADOW_INSET};
    }}
    .rate-up {{
        color: {SOFT_TEAL};
        background: rgba(74, 222, 128, 0.09);
    }}
    .rate-down {{
        color: {SOFT_ROSE};
        background: rgba(251, 113, 133, 0.10);
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
        color_continuous_scale=[SOFT_SURFACE, SOFT_ACCENT_LIGHT, SOFT_ACCENT],
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
            show_text_metric(f"{cluster_name}", f"{peak_hour:02d}시", f"피크 {peak_share:.1f}%")
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

    render_pretty_table(pd.DataFrame(display_rows), "K-means 군집별 시간대 패턴 요약(%)", max_rows=10)


def monthly_tab(monthly_df: pd.DataFrame, top_n: int) -> None:
    """장기 월별 추세 분석 탭을 구성합니다."""
    if monthly_df.empty or not {"year", "month"}.issubset(monthly_df.columns):
        empty_message("월별 장기 데이터가 없어 장기 추세 분석을 표시할 수 없습니다.")
        return

    render_section_header(
        "Longitudinal View",
        "장기 월별 추세",
        "월별 이용량의 변화를 전년 동월 기준으로 비교하고, 급증·급감 정류소와 시간대 군집 패턴을 함께 봅니다.",
    )

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

    render_section_header(
        "Weather Gallery",
        "날씨와 버스 이용",
        "월별 날씨 변수와 버스 이용량을 결합해 온도, 강수량, 계절 변화와 이용 패턴의 동행성을 탐색합니다.",
    )

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

    col1, col2, col3 = st.columns([1.1, 1.1, 1], gap="large")
    with col1:
        weather_col = st.selectbox(
            "날씨 변수 선택",
            list(weather_options.keys()),
            format_func=lambda col: weather_options[col],
        )
    with col2:
        stop_options = ["전체 대구"]
        if "stop_name" in monthly_df.columns:
            stop_options += sorted(monthly_df["stop_name"].dropna().astype(str).unique())
        selected_stop = st.selectbox("전체 대구 또는 정류소 선택", stop_options)
    with col3:
        st.markdown("<div class='weather-metric-align-spacer'></div>", unsafe_allow_html=True)
        show_metric("날씨 관측 월 수", len(weather_monthly))

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
        show_metric("Pearson", selected_corr.get("Pearson 상관계수"), decimals=2)
    with metric_cols[1]:
        show_metric("Spearman", selected_corr.get("Spearman 상관계수"), decimals=2)
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

    selected_scatter = plot_weather_scatter(
        filtered,
        weather_col,
        weather_options[weather_col],
        f"{weather_options[weather_col]}와 월별 승차 인원의 산점도",
    )
    render_two_column_charts(
        [
            (
                plot_weather_monthly_line(filtered, "boardings", "월별 승차 인원", "월별 버스 승차 인원 추세"),
                "월별 버스 승차 인원 추세를 표시할 수 없습니다.",
            ),
            (
                plot_weather_monthly_line(filtered, "monthly_avg_temp", "월평균 기온", "월별 기온 추세"),
                "월평균 기온 컬럼이 없어 기온 추세를 표시할 수 없습니다.",
            ),
            (
                plot_weather_bus_dual_line(filtered, weather_col, weather_options[weather_col]),
                "버스 이용량과 날씨 추세 비교 그래프를 표시할 수 없습니다.",
            ),
            (
                selected_scatter,
                "선택한 날씨 변수 산점도를 표시할 수 없습니다.",
            ),
        ]
    )

    render_two_column_charts(
        [
            (
                plot_weather_scatter(
                    filtered,
                    "monthly_avg_temp",
                    "월평균 기온",
                    "기온과 월별 승차 인원의 산점도",
                ),
                "기온 산점도를 표시할 수 없습니다.",
            ),
            (
                plot_weather_scatter(
                    filtered,
                    "monthly_precipitation",
                    "월 누적 강수량",
                    "강수량과 월별 승차 인원의 산점도",
                ),
                "강수량 산점도를 표시할 수 없습니다.",
            ),
            (
                plot_season_bus_box(filtered),
                "계절별 이용량 박스플롯을 표시할 수 없습니다.",
            ),
            (
                plot_weather_correlation_heatmap(correlation_df),
                "날씨 변수 상관관계 히트맵을 표시할 수 없습니다.",
            ),
        ]
    )

    st.subheader("날씨 변수별 상관계수")
    display_corr = correlation_df.drop(columns=["weather_column"], errors="ignore").copy()
    for col in ["Pearson 상관계수", "Spearman 상관계수"]:
        if col in display_corr.columns:
            display_corr[col] = display_corr[col].round(2)
    render_pretty_table(display_corr, "날씨 변수별 상관계수", max_rows=12)

    st.subheader("날씨 분석에서 함께 확인한 사항")
    st.markdown(
        """
- 공휴일과 방학이 월별 이용량을 함께 움직일 수 있다는 점을 확인했습니다.
- 노선 개편과 배차 변화가 날씨와 별도로 이용량 변화를 만들 수 있다는 점을 확인했습니다.
- 지역 행사와 대형 집객 시설 일정이 월별 수요 변화와 함께 나타날 수 있습니다.
- 유가 변화와 대체 교통수단 이용 변화도 버스 이용량 해석에 함께 고려할 조건입니다.
- 코로나19와 같은 외부 요인은 장기 추세를 크게 바꿀 수 있는 배경 조건입니다.
- 월별 자료에서는 개별 강수일의 즉각적인 영향보다 월 단위 경향을 확인하는 데 적합합니다.
"""
    )


def data_limit_tab(bundle: dict) -> None:
    """데이터 해석 전에 함께 확인한 참고사항을 표시합니다."""
    render_section_header(
        "Method Notes",
        "데이터 해석 참고사항",
        "분석 결과를 더 정확하게 읽기 위해 함께 확인한 데이터 수집 조건과 해석 기준입니다.",
    )
    st.subheader("분석에서 함께 확인한 사항")
    st.markdown(
        """
- 하차 태그를 하지 않은 승객이 있을 수 있어 하차 인원은 보조 지표로 함께 확인했습니다.
- 현금 승차가 데이터에 포함되지 않을 수 있어 승차 인원 해석 시 집계 범위를 함께 확인했습니다.
- 경유 노선 수와 실제 배차 횟수는 다를 수 있어 노선 수는 공급 규모의 1차 지표로 사용했습니다.
- 버스 배차 간격과 차량 크기 자료가 추가되면 후보 정류소의 검토 우선순위를 더 정교하게 만들 수 있습니다.
- 이용객 수는 실제 차량 내부 혼잡도를 직접 확정하기보다 혼잡 가능성을 살피는 출발점으로 사용했습니다.
- 특정 정류소의 높은 이용량은 주변 학교, 상권, 병원, 환승센터 등 지역 맥락과 함께 해석했습니다.
- 분석 결과는 노선 부족 확정이 아니라 추가 검토 후보를 좁히는 데 활용했습니다.
- 데이터 수집 기간에 따라 계절성과 일시적 이벤트가 함께 나타날 수 있다는 점을 확인했습니다.
- 공휴일과 방학, 노선 개편, 지역 행사, 유가 변화, 코로나19 등의 외부 요인도 이용량 해석 조건으로 확인했습니다.
- 월별 자료는 개별 강수일의 즉각적 영향보다 월 단위 흐름을 확인하는 데 적합합니다.
"""
    )


def main() -> None:
    """Streamlit 앱의 시작점입니다."""
    st.set_page_config(page_title="대구 시내버스 수급 불균형 분석", layout="wide")
    ensure_directories(BASE_DIR)
    set_korean_font()
    apply_retro_90s_overrides()
    render_app_hero()
    inject_heart_easter_egg()

    if st.sidebar.button("정제 데이터 다시 불러오기"):
        bundle = reload_project_data()
        st.sidebar.success("정제 CSV를 다시 불러왔습니다.")
    else:
        bundle = get_project_data()
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

    analysis_tables = build_core_analysis_tables(
        filtered_stop,
        filtered_hourly,
        monthly_df,
        top_n=filters["top_n"],
    )

    with st.expander("핵심 분석 결과 표 보기"):
        render_analysis_tables_grid(analysis_tables, max_rows=12)

    tabs = st.tabs(
        [
            "전체 현황",
            "시간대별 분석",
            "정류소별 분석",
            "수요·공급 불균형 분석",
            "지도 분석",
            "장기 추세 분석",
            "날씨와 버스 이용",
            "데이터 해석 참고사항",
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

    render_marquee_footer()


if __name__ == "__main__":
    main()
