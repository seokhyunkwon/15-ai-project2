from pathlib import Path

import numpy as np
import pandas as pd

from src.column_mapper import find_best_column, normalize_column_name
from src.preprocessing import to_numeric_series


WEATHER_COLUMN_CANDIDATES = {
    "date": ["년월", "연월", "날짜", "일자", "일시", "년월일", "date", "datetime", "time"],
    "avg_temp": ["월평균기온", "평균기온", "평균 기온", "평균기온(°C)", "avg_temp", "tavg"],
    "max_temp": ["월평균최고기온", "평균최고기온", "평균 최고기온", "평균최고기온(°C)", "최고기온", "최고 기온", "최고기온(°C)", "max_temp", "tmax"],
    "min_temp": ["월평균최저기온", "평균최저기온", "평균 최저기온", "평균최저기온(°C)", "최저기온", "최저 기온", "최저기온(°C)", "min_temp", "tmin"],
    "precipitation": ["월합강수량", "월누적강수량", "월 강수량", "월합강수량(00~24h만)(mm)", "일강수량", "강수량", "일강수량(mm)", "rain", "precipitation"],
    "rainy_days": ["월강수일수", "강수일수", "비온날수", "rainy_days", "rain_days"],
    "humidity": ["월평균습도", "평균습도", "평균상대습도", "평균 상대습도", "평균 상대습도(%)", "humidity"],
    "wind_speed": ["월평균풍속", "평균풍속", "평균 풍속", "평균 풍속(m/s)", "wind_speed"],
    "sunshine_hours": ["월총일조시간", "총일조시간", "일조시간", "합계 일조시간", "합계 일조시간(hr)", "sunshine"],
    "heatwave_days": ["폭염일수", "폭염 일수", "heatwave_days"],
    "coldwave_days": ["한파일수", "한파 일수", "coldwave_days"],
    "snowfall": ["월최심적설", "최심적설", "적설량", "일 최심적설", "일 최심적설(cm)", "snowfall"],
}


WEATHER_LABELS = {
    "monthly_avg_temp": "월평균 기온",
    "monthly_avg_max_temp": "월평균 최고기온",
    "monthly_avg_min_temp": "월평균 최저기온",
    "monthly_precipitation": "월 누적 강수량",
    "rainy_days": "월 강수일수",
    "monthly_avg_humidity": "월평균 습도",
    "monthly_avg_wind_speed": "월평균 풍속",
    "monthly_sunshine_hours": "월 총 일조시간",
    "heatwave_days": "폭염일수",
    "coldwave_days": "한파일수",
    "monthly_snowfall": "월 최심적설",
}


def map_weather_columns(columns) -> dict:
    """날씨 CSV의 실제 컬럼명을 분석용 표준 컬럼명으로 연결합니다."""
    mapping = {}
    for standard_name, candidates in WEATHER_COLUMN_CANDIDATES.items():
        mapping[standard_name] = find_best_column(columns, candidates)
    return mapping


def looks_like_weather_file(df: pd.DataFrame, file_name: str) -> bool:
    """파일명과 컬럼 구조를 보고 날씨 데이터인지 판단합니다."""
    mapping = map_weather_columns(df.columns)
    normalized_name = normalize_column_name(file_name)
    has_weather_columns = bool(mapping.get("date")) and any(
        mapping.get(key) for key in ["avg_temp", "max_temp", "min_temp", "precipitation", "humidity"]
    )
    has_weather_name = any(token in normalized_name for token in ["asos", "weather", "기상", "날씨"])
    return has_weather_columns and (has_weather_name or "지점명" in df.columns)


def parse_weather_dates(series: pd.Series) -> pd.Series:
    """날씨 날짜 컬럼을 일자 또는 연월 형태 모두 처리해 datetime으로 바꿉니다."""
    text = series.astype(str).str.strip()
    parsed = pd.to_datetime(text, errors="coerce")
    compact_month = parsed.isna() & text.str.match(r"^\d{6}$", na=False)
    if compact_month.any():
        parsed.loc[compact_month] = pd.to_datetime(text.loc[compact_month], format="%Y%m", errors="coerce")
    return parsed


def is_monthly_weather_dataframe(df: pd.DataFrame, mapping: dict, file_name: str) -> bool:
    """날씨 파일이 이미 월별 자료인지 판단합니다."""
    date_col = mapping.get("date")
    if not date_col or date_col not in df.columns:
        return False
    normalized_name = normalize_column_name(file_name)
    if "mnh" in normalized_name or "월" in normalized_name:
        return True
    sample = df[date_col].dropna().astype(str).str.strip().head(50)
    if sample.empty:
        return False
    month_like = sample.str.match(r"^\d{4}[-./]?\d{1,2}$|^\d{6}$", na=False).mean()
    monthly_column_hint = any("월합" in str(col) or "평균최고" in str(col) or "평균최저" in str(col) for col in df.columns)
    return bool(month_like >= 0.8 or monthly_column_hint)


def find_weather_payloads(loaded_files: dict[str, dict]) -> list[dict]:
    """불러온 CSV 묶음에서 날씨 파일만 골라냅니다."""
    weather_payloads = []
    for file_name, payload in loaded_files.items():
        df = payload.get("data", pd.DataFrame())
        if isinstance(df, pd.DataFrame) and looks_like_weather_file(df, file_name):
            weather_payloads.append({**payload, "file_name": file_name})
    return weather_payloads


def standardize_weather_dataframe(df: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, dict]:
    """일별 날씨 원본을 날짜와 주요 기상 지표 중심의 표준 데이터로 바꿉니다."""
    mapping = map_weather_columns(df.columns)
    if not mapping.get("date"):
        return pd.DataFrame(), mapping

    result = pd.DataFrame(index=df.index)
    result["source_file"] = source_file
    result["date"] = parse_weather_dates(df[mapping["date"]])

    for standard_col in [
        "avg_temp",
        "max_temp",
        "min_temp",
        "precipitation",
        "humidity",
        "wind_speed",
        "sunshine_hours",
        "snowfall",
    ]:
        actual_col = mapping.get(standard_col)
        if actual_col and actual_col in df.columns:
            result[standard_col] = to_numeric_series(df[actual_col])

    result = result[result["date"].notna()].copy()
    if result.empty:
        return result, mapping

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["period"] = result["date"].dt.to_period("M").astype(str)

    # 기상청 일자료에서 강수량과 적설량 공란은 대체로 관측 없음에 가깝기 때문에 합계용으로만 0 처리합니다.
    if "precipitation" in result.columns:
        result["precipitation_for_sum"] = result["precipitation"].fillna(0)
    if "snowfall" in result.columns:
        result["snowfall_for_sum"] = result["snowfall"].fillna(0)
    return result, mapping


def standardize_monthly_weather_dataframe(df: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, dict]:
    """월별 날씨 원본을 앱에서 쓰는 월별 표준 컬럼으로 바로 바꿉니다."""
    mapping = map_weather_columns(df.columns)
    if not mapping.get("date"):
        return pd.DataFrame(), mapping

    result = pd.DataFrame(index=df.index)
    result["source_file"] = source_file
    result["date"] = parse_weather_dates(df[mapping["date"]])
    result = result[result["date"].notna()].copy()
    if result.empty:
        return result, mapping

    result["year"] = result["date"].dt.year
    result["month"] = result["date"].dt.month
    result["period"] = result["date"].dt.to_period("M").astype(str)
    result["season"] = result["month"].apply(month_to_season)

    monthly_targets = {
        "avg_temp": "monthly_avg_temp",
        "max_temp": "monthly_avg_max_temp",
        "min_temp": "monthly_avg_min_temp",
        "precipitation": "monthly_precipitation",
        "rainy_days": "rainy_days",
        "humidity": "monthly_avg_humidity",
        "wind_speed": "monthly_avg_wind_speed",
        "sunshine_hours": "monthly_sunshine_hours",
        "heatwave_days": "heatwave_days",
        "coldwave_days": "coldwave_days",
        "snowfall": "monthly_snowfall",
    }
    for standard_col, target_col in monthly_targets.items():
        actual_col = mapping.get(standard_col)
        if actual_col and actual_col in df.columns:
            result[target_col] = to_numeric_series(df.loc[result.index, actual_col])

    # 기상청 월자료의 최심적설 공란은 적설 관측이 없는 달로 해석되는 경우가 많아 이 변수만 0으로 보정합니다.
    if "monthly_snowfall" in result.columns:
        result["monthly_snowfall"] = result["monthly_snowfall"].fillna(0)

    return result, mapping


def aggregate_weather_monthly(weather_daily: pd.DataFrame) -> pd.DataFrame:
    """일별 날씨를 버스 월별 데이터와 결합할 수 있는 월별 지표로 집계합니다."""
    if weather_daily.empty or not {"year", "month"}.issubset(weather_daily.columns):
        return pd.DataFrame()

    grouped = weather_daily.groupby(["year", "month"], as_index=False)
    monthly = grouped.size().rename(columns={"size": "weather_days"})

    if "avg_temp" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["avg_temp"].mean().rename(columns={"avg_temp": "monthly_avg_temp"}),
            on=["year", "month"],
            how="left",
        )
    if "max_temp" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["max_temp"].mean().rename(columns={"max_temp": "monthly_avg_max_temp"}),
            on=["year", "month"],
            how="left",
        )
        heatwave = (
            weather_daily.assign(heatwave_day=weather_daily["max_temp"] >= 33)
            .groupby(["year", "month"], as_index=False)["heatwave_day"]
            .sum()
            .rename(columns={"heatwave_day": "heatwave_days"})
        )
        monthly = monthly.merge(heatwave, on=["year", "month"], how="left")
    if "min_temp" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["min_temp"].mean().rename(columns={"min_temp": "monthly_avg_min_temp"}),
            on=["year", "month"],
            how="left",
        )
        coldwave = (
            weather_daily.assign(coldwave_day=weather_daily["min_temp"] <= -12)
            .groupby(["year", "month"], as_index=False)["coldwave_day"]
            .sum()
            .rename(columns={"coldwave_day": "coldwave_days"})
        )
        monthly = monthly.merge(coldwave, on=["year", "month"], how="left")
    if "precipitation_for_sum" in weather_daily.columns:
        precip = (
            weather_daily.groupby(["year", "month"], as_index=False)["precipitation_for_sum"]
            .sum()
            .rename(columns={"precipitation_for_sum": "monthly_precipitation"})
        )
        rainy_days = (
            weather_daily.assign(rainy_day=weather_daily["precipitation_for_sum"] > 0)
            .groupby(["year", "month"], as_index=False)["rainy_day"]
            .sum()
            .rename(columns={"rainy_day": "rainy_days"})
        )
        monthly = monthly.merge(precip, on=["year", "month"], how="left")
        monthly = monthly.merge(rainy_days, on=["year", "month"], how="left")
    if "humidity" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["humidity"].mean().rename(columns={"humidity": "monthly_avg_humidity"}),
            on=["year", "month"],
            how="left",
        )
    if "wind_speed" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["wind_speed"].mean().rename(columns={"wind_speed": "monthly_avg_wind_speed"}),
            on=["year", "month"],
            how="left",
        )
    if "sunshine_hours" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["sunshine_hours"].sum(min_count=1).rename(columns={"sunshine_hours": "monthly_sunshine_hours"}),
            on=["year", "month"],
            how="left",
        )
    if "snowfall_for_sum" in weather_daily.columns:
        monthly = monthly.merge(
            grouped["snowfall_for_sum"].sum().rename(columns={"snowfall_for_sum": "monthly_snowfall"}),
            on=["year", "month"],
            how="left",
        )

    monthly["period"] = monthly["year"].astype(int).astype(str) + "-" + monthly["month"].astype(int).astype(str).str.zfill(2)
    monthly["season"] = monthly["month"].apply(month_to_season)
    return monthly.sort_values(["year", "month"]).reset_index(drop=True)


def combine_weather_monthly_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """여러 월별 날씨 표를 한 달 한 행으로 합칩니다."""
    valid_frames = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    if not valid_frames:
        return pd.DataFrame()

    data = pd.concat(valid_frames, ignore_index=True, sort=False)
    data = data.dropna(subset=["year", "month"]).copy()
    if data.empty:
        return pd.DataFrame()

    numeric_cols = [
        col
        for col in data.columns
        if col not in ["source_file", "date", "period", "season", "year", "month"]
    ]
    for col in numeric_cols:
        data[col] = to_numeric_series(data[col])

    agg_dict = {col: "mean" for col in numeric_cols}
    if "source_file" in data.columns:
        agg_dict["source_file"] = lambda values: ", ".join(sorted(set(values.dropna().astype(str))))

    monthly = data.groupby(["year", "month"], as_index=False).agg(agg_dict)
    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)
    monthly["period"] = monthly["year"].astype(str) + "-" + monthly["month"].astype(str).str.zfill(2)
    monthly["season"] = monthly["month"].apply(month_to_season)
    return monthly.sort_values(["year", "month"]).reset_index(drop=True)


def month_to_season(month: int) -> str:
    """월 숫자를 계절명으로 바꿉니다."""
    if month in [3, 4, 5]:
        return "봄"
    if month in [6, 7, 8]:
        return "여름"
    if month in [9, 10, 11]:
        return "가을"
    return "겨울"


def inspect_weather_payloads(weather_payloads: list[dict]) -> pd.DataFrame:
    """날씨 파일의 인코딩, 크기, 컬럼, 날짜 범위, 결측값을 표로 정리합니다."""
    rows = []
    for payload in weather_payloads:
        df = payload.get("data", pd.DataFrame())
        file_name = payload.get("file_name", "")
        encoding = payload.get("encoding", "")
        mapping = map_weather_columns(df.columns) if isinstance(df, pd.DataFrame) else {}
        date_range = "-"
        if mapping.get("date") and mapping["date"] in df.columns:
            dates = pd.to_datetime(df[mapping["date"]], errors="coerce")
            if dates.notna().any():
                date_range = f"{dates.min().date()} ~ {dates.max().date()}"
        rows.append(
            {
                "파일명": file_name,
                "인코딩": encoding,
                "행 수": int(df.shape[0]),
                "열 수": int(df.shape[1]),
                "컬럼명": ", ".join(map(str, df.columns)),
                "데이터 타입": str({str(col): str(dtype) for col, dtype in df.dtypes.items()}),
                "날짜 범위": date_range,
                "결측값": str({str(col): int(count) for col, count in df.isna().sum().items()}),
            }
        )
    return pd.DataFrame(rows)


def prepare_weather_bundle(loaded_files: dict[str, dict]) -> dict:
    """날씨 파일 탐지부터 월별 표준화까지 한 번에 수행합니다."""
    weather_payloads = find_weather_payloads(loaded_files)
    daily_frames = []
    monthly_frames = []
    mapping_rows = []

    for payload in weather_payloads:
        file_name = payload.get("file_name", "")
        df = payload.get("data", pd.DataFrame())
        mapping = map_weather_columns(df.columns) if isinstance(df, pd.DataFrame) else {}
        if isinstance(df, pd.DataFrame) and is_monthly_weather_dataframe(df, mapping, file_name):
            monthly, mapping = standardize_monthly_weather_dataframe(df, file_name)
            if not monthly.empty:
                monthly_frames.append(monthly)
            time_unit = "monthly"
        else:
            daily, mapping = standardize_weather_dataframe(df, file_name)
            if not daily.empty:
                daily_frames.append(daily)
            time_unit = "daily"
        for standard_col, actual_col in mapping.items():
            mapping_rows.append(
                {
                    "file_name": file_name,
                    "time_unit": time_unit,
                    "standard_column": standard_col,
                    "actual_column": actual_col,
                }
            )

    weather_daily = (
        pd.concat(daily_frames, ignore_index=True, sort=False)
        if daily_frames
        else pd.DataFrame(columns=["source_file", "date", "year", "month", "period"])
    )
    if not weather_daily.empty:
        weather_daily = weather_daily.drop_duplicates(subset=["source_file", "date"]).sort_values("date")
    monthly_from_daily = aggregate_weather_monthly(weather_daily)
    all_monthly_frames = monthly_frames + ([monthly_from_daily] if not monthly_from_daily.empty else [])
    weather_monthly = combine_weather_monthly_frames(all_monthly_frames)

    return {
        "weather_files": weather_payloads,
        "weather_audit": inspect_weather_payloads(weather_payloads),
        "weather_mapping": pd.DataFrame(mapping_rows),
        "weather_daily": weather_daily,
        "weather_monthly": weather_monthly,
    }


def infer_bus_time_unit(monthly_df: pd.DataFrame, hourly_df: pd.DataFrame) -> str:
    """버스 데이터가 어떤 시간 단위로 분석 가능한지 설명합니다."""
    has_monthly = not monthly_df.empty and {"year", "month"}.issubset(monthly_df.columns)
    has_hourly = not hourly_df.empty and "hour" in hourly_df.columns
    if has_monthly and has_hourly:
        return "월별 정류소 승하차 데이터와 월별 시간대 합계 데이터가 있습니다."
    if has_monthly:
        return "월별 정류소 승하차 데이터가 있습니다."
    if has_hourly:
        return "시간대별 데이터는 있으나 월 정보가 집계 과정에서 사라져 날씨 월자료와 직접 결합하기 어렵습니다."
    return "날씨와 결합할 수 있는 날짜 또는 월별 버스 데이터가 없습니다."


def aggregate_bus_monthly(monthly_df: pd.DataFrame, stop_name: str | None = None) -> pd.DataFrame:
    """정류소 선택에 따라 버스 월별 이용량을 한 달 한 행으로 집계합니다."""
    if monthly_df.empty or not {"year", "month"}.issubset(monthly_df.columns):
        return pd.DataFrame()

    metric_cols = [col for col in ["boardings", "alightings", "passengers"] if col in monthly_df.columns]
    if not metric_cols:
        return pd.DataFrame()

    data = monthly_df.copy()
    if stop_name and stop_name != "전체 대구":
        if "stop_name" not in data.columns:
            return pd.DataFrame()
        data = data[data["stop_name"].astype(str) == str(stop_name)].copy()
    if data.empty:
        return pd.DataFrame()

    grouped = data.groupby(["year", "month"], as_index=False)[metric_cols].sum()
    if "boardings" not in grouped.columns and "passengers" in grouped.columns:
        grouped["boardings"] = grouped["passengers"]
    if "alightings" in grouped.columns:
        grouped["total_riders"] = grouped["boardings"].fillna(0) + grouped["alightings"].fillna(0)
    grouped["period"] = grouped["year"].astype(int).astype(str) + "-" + grouped["month"].astype(int).astype(str).str.zfill(2)
    grouped["season"] = grouped["month"].apply(month_to_season)
    return grouped.sort_values(["year", "month"]).reset_index(drop=True)


def merge_bus_weather_monthly(bus_monthly: pd.DataFrame, weather_monthly: pd.DataFrame) -> pd.DataFrame:
    """월별 버스 이용량과 월별 날씨 지표를 한 달 한 행으로 결합합니다."""
    if bus_monthly.empty or weather_monthly.empty:
        return pd.DataFrame()
    merged = bus_monthly.merge(weather_monthly, on=["year", "month", "period", "season"], how="inner")
    return merged.sort_values(["year", "month"]).reset_index(drop=True)


def available_weather_variables(weather_monthly: pd.DataFrame) -> dict:
    """월별 날씨 데이터에 실제로 존재하는 변수만 선택지로 제공합니다."""
    if weather_monthly.empty:
        return {}
    return {col: label for col, label in WEATHER_LABELS.items() if col in weather_monthly.columns and weather_monthly[col].notna().any()}


def calculate_weather_correlations(merged_df: pd.DataFrame, bus_metric: str = "boardings") -> pd.DataFrame:
    """버스 이용량과 날씨 변수의 Pearson, Spearman 상관계수를 계산합니다."""
    if merged_df.empty or bus_metric not in merged_df.columns:
        return pd.DataFrame()
    rows = []
    for weather_col, label in available_weather_variables(merged_df).items():
        data = merged_df[[bus_metric, weather_col]].dropna()
        if len(data) < 3 or data[bus_metric].nunique() < 2 or data[weather_col].nunique() < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = data[bus_metric].corr(data[weather_col], method="pearson")
            spearman = data[bus_metric].corr(data[weather_col], method="spearman")
        rows.append(
            {
                "날씨 변수": label,
                "weather_column": weather_col,
                "관측 월 수": len(data),
                "Pearson 상관계수": pearson,
                "Spearman 상관계수": spearman,
                "해석": interpret_weather_correlation(pearson, label),
            }
        )
    return pd.DataFrame(rows)


def interpret_weather_correlation(value: float, weather_label: str) -> str:
    """상관계수를 인과가 아닌 경향 문장으로 해석합니다."""
    if pd.isna(value):
        return f"{weather_label}와 버스 이용량의 상관관계를 계산할 관측치가 부족합니다."
    strength = abs(value)
    if strength < 0.2:
        strength_text = "약한"
    elif strength < 0.4:
        strength_text = "다소 약한"
    elif strength < 0.6:
        strength_text = "보통"
    elif strength < 0.8:
        strength_text = "강한"
    else:
        strength_text = "매우 강한"
    direction = "증가하는" if value > 0 else "감소하는"
    particle = korean_subject_particle(weather_label)
    return f"{weather_label}{particle} 높은 달에 버스 이용량이 {direction} 경향이 관찰되었습니다. 상관관계는 {strength_text} 수준이며 인과관계를 의미하지 않습니다."


def korean_subject_particle(text: str) -> str:
    """한글 마지막 글자의 받침 여부에 따라 이/가 조사를 고릅니다."""
    if not text:
        return "이"
    last_char = text[-1]
    code = ord(last_char)
    if 0xAC00 <= code <= 0xD7A3:
        return "이" if (code - 0xAC00) % 28 else "가"
    return "이"


def selected_weather_correlation_summary(correlation_df: pd.DataFrame, weather_col: str) -> str:
    """선택한 날씨 변수의 상관분석 결과를 한 문장으로 반환합니다."""
    if correlation_df.empty:
        return "상관관계를 계산할 수 없습니다."
    row = correlation_df[correlation_df["weather_column"] == weather_col]
    if row.empty:
        return "선택한 날씨 변수의 상관관계를 계산할 수 없습니다."
    item = row.iloc[0]
    pearson = item["Pearson 상관계수"]
    spearman = item["Spearman 상관계수"]
    if pd.isna(pearson):
        return item["해석"]
    return (
        f"{item['날씨 변수']}와 월별 승차 인원의 Pearson 상관계수는 {pearson:.2f}, "
        f"Spearman 상관계수는 {spearman:.2f}입니다. {item['해석']}"
    )
