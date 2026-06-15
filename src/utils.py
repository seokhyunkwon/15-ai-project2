from pathlib import Path

import numpy as np
import pandas as pd


def ensure_directories(base_dir: str | Path) -> None:
    """프로젝트 실행에 필요한 폴더를 만듭니다."""
    base = Path(base_dir)
    for relative in ["data", "src", "outputs", "outputs/figures", "outputs/processed"]:
        (base / relative).mkdir(parents=True, exist_ok=True)


def set_korean_font() -> str | None:
    """Matplotlib과 seaborn 그래프에서 한글이 깨지지 않도록 폰트를 설정합니다."""
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm

    preferred_fonts = [
        "Malgun Gothic",
        "맑은 고딕",
        "NanumGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "AppleGothic",
    ]
    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next((font for font in preferred_fonts if font in installed), None)
    if selected:
        plt.rcParams["font.family"] = selected
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def format_number(value, decimals: int = 0) -> str:
    """숫자를 천 단위 쉼표가 있는 문자열로 바꿉니다."""
    if value is None:
        return "-"
    try:
        if pd.isna(value):
            return "-"
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}"


def safe_divide(numerator, denominator, fill_value=np.nan):
    """0으로 나누는 오류를 막으면서 나눗셈을 수행합니다."""
    numerator = pd.Series(numerator)
    denominator = pd.Series(denominator)
    result = numerator / denominator.replace({0: np.nan})
    return result.replace([np.inf, -np.inf], np.nan).fillna(fill_value)


def save_dataframe_csv(df: pd.DataFrame, path: str | Path) -> None:
    """데이터프레임을 엑셀에서 열기 쉬운 utf-8-sig CSV로 저장합니다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.DataFrame()
    df.to_csv(path, index=False, encoding="utf-8-sig")


def make_download_csv(df: pd.DataFrame) -> bytes:
    """Streamlit 다운로드 버튼에 넣을 CSV 바이트를 만듭니다."""
    if df is None:
        df = pd.DataFrame()
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def first_non_null(series: pd.Series):
    """그룹 안에서 처음으로 비어 있지 않은 값을 찾습니다."""
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else np.nan
