from pathlib import Path

import numpy as np
import pandas as pd

from src.column_mapper import (
    find_time_columns,
    infer_roles,
    infer_time_metric_from_column,
    map_columns,
    parse_hour,
)
from src.utils import first_non_null, safe_divide


COUNT_COLUMNS = ["boardings", "alightings", "passengers", "route_count"]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 앞뒤 공백을 제거하고 중복 컬럼명에는 번호를 붙입니다."""
    cleaned = df.copy()
    raw_columns = [str(col).strip() for col in cleaned.columns]
    seen = {}
    new_columns = []
    for col in raw_columns:
        if col not in seen:
            seen[col] = 0
            new_columns.append(col)
        else:
            seen[col] += 1
            new_columns.append(f"{col}_{seen[col]}")
    cleaned.columns = new_columns
    return cleaned


def to_numeric_series(series: pd.Series) -> pd.Series:
    """쉼표가 포함된 숫자 문자열을 숫자형으로 변환합니다."""
    if series is None:
        return pd.Series(dtype=float)
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_stop_id(series: pd.Series) -> pd.Series:
    """정류소 ID를 문자열로 통일합니다."""
    normalized = series.astype(str).str.strip()
    normalized = normalized.str.replace(r"\.0+$", "", regex=True)
    return normalized.replace({"nan": pd.NA, "None": pd.NA, "": pd.NA, "<NA>": pd.NA})


def create_merge_key(df: pd.DataFrame) -> pd.Series:
    """정류소 ID가 있으면 ID를, 없으면 정류소명과 행정구역을 병합 키로 사용합니다."""
    stop_id = df["stop_id"] if "stop_id" in df.columns else pd.Series(pd.NA, index=df.index)
    stop_name = df["stop_name"] if "stop_name" in df.columns else pd.Series("", index=df.index)
    district = df["district"] if "district" in df.columns else pd.Series("", index=df.index)
    id_key = "id:" + stop_id.astype(str)
    name_key = "name:" + stop_name.astype(str).str.strip() + "|" + district.astype(str).str.strip()
    return np.where(stop_id.notna() & (stop_id.astype(str).str.strip() != ""), id_key, name_key)


def clean_lat_lon(df: pd.DataFrame) -> pd.DataFrame:
    """대한민국 범위를 크게 벗어나는 위도·경도 이상값을 제거합니다."""
    cleaned = df.copy()
    if "lat" in cleaned.columns:
        cleaned["lat"] = to_numeric_series(cleaned["lat"])
        cleaned.loc[~cleaned["lat"].between(33, 39), "lat"] = np.nan
    if "lon" in cleaned.columns:
        cleaned["lon"] = to_numeric_series(cleaned["lon"])
        cleaned.loc[~cleaned["lon"].between(124, 132), "lon"] = np.nan
    return cleaned


def remove_negative_counts(df: pd.DataFrame) -> pd.DataFrame:
    """승차·하차·이용객 수가 음수인 행을 제거합니다."""
    cleaned = df.copy()
    count_cols = [col for col in ["boardings", "alightings", "passengers"] if col in cleaned.columns]
    for col in count_cols:
        cleaned[col] = to_numeric_series(cleaned[col])
    if not count_cols:
        return cleaned
    mask = pd.Series(True, index=cleaned.index)
    for col in count_cols:
        mask &= cleaned[col].isna() | (cleaned[col] >= 0)
    return cleaned[mask].copy()


def standardize_long_dataframe(df: pd.DataFrame, mapping: dict, source_name: str) -> pd.DataFrame:
    """long 형식 또는 일반 표 형식 데이터를 표준 컬럼명으로 변환합니다."""
    standardized = pd.DataFrame(index=df.index)
    standardized["source_file"] = source_name

    for standard_col, actual_col in mapping.items():
        if actual_col and actual_col in df.columns:
            standardized[standard_col] = df[actual_col]

    if "stop_id" in standardized.columns:
        standardized["stop_id"] = normalize_stop_id(standardized["stop_id"])
    if "hour" in standardized.columns:
        standardized["hour"] = standardized["hour"].apply(parse_hour)
    if "date" in standardized.columns:
        standardized["date"] = pd.to_datetime(standardized["date"], errors="coerce")
        standardized["year"] = standardized.get("year", standardized["date"].dt.year)
        standardized["month"] = standardized.get("month", standardized["date"].dt.month)
    if "year" in standardized.columns:
        standardized["year"] = to_numeric_series(standardized["year"])
    if "month" in standardized.columns:
        standardized["month"] = to_numeric_series(standardized["month"])

    for col in COUNT_COLUMNS:
        if col in standardized.columns:
            standardized[col] = to_numeric_series(standardized[col])

    standardized = clean_lat_lon(standardized)
    standardized = remove_negative_counts(standardized)
    standardized = standardized.drop_duplicates().copy()
    standardized["_merge_key"] = create_merge_key(standardized)
    return standardized


def melt_wide_time_dataframe(df: pd.DataFrame, mapping: dict, source_name: str) -> pd.DataFrame:
    """05시, 06시처럼 펼쳐진 시간대 컬럼을 long 형식으로 변환합니다."""
    time_info = find_time_columns(df.columns, mapping.values())
    if not time_info:
        return pd.DataFrame()

    time_cols = [item["column"] for item in time_info]
    id_cols = [col for col in df.columns if col not in time_cols]
    melted = df.melt(id_vars=id_cols, value_vars=time_cols, var_name="time_column", value_name="count_value")
    melted["hour"] = melted["time_column"].apply(parse_hour)
    metric_type_col = mapping.get("metric_type")
    if metric_type_col and metric_type_col in melted.columns:
        melted["metric"] = melted[metric_type_col].apply(lambda value: infer_time_metric_from_column(value, ""))
    else:
        melted["metric"] = melted["time_column"].apply(lambda col: infer_time_metric_from_column(col, source_name))
    melted["count_value"] = to_numeric_series(melted["count_value"])
    melted = melted[melted["count_value"].notna() & (melted["count_value"] >= 0)].copy()

    base_mapping = {
        key: value
        for key, value in mapping.items()
        if key not in ["boardings", "alightings", "passengers", "hour", "metric_type"]
    }
    base = pd.DataFrame(index=melted.index)
    base["source_file"] = source_name
    for standard_col, actual_col in base_mapping.items():
        if actual_col and actual_col in melted.columns:
            base[standard_col] = melted[actual_col]
    if "stop_id" in base.columns:
        base["stop_id"] = normalize_stop_id(base["stop_id"])
    if "date" in base.columns:
        base["date"] = pd.to_datetime(base["date"], errors="coerce")
        base["year"] = base.get("year", base["date"].dt.year)
        base["month"] = base.get("month", base["date"].dt.month)
    if "year" in base.columns:
        base["year"] = to_numeric_series(base["year"])
    if "month" in base.columns:
        base["month"] = to_numeric_series(base["month"])
    if "route_count" in base.columns:
        base["route_count"] = to_numeric_series(base["route_count"])
    base = clean_lat_lon(base)
    base["_merge_key"] = create_merge_key(base)
    base["hour"] = melted["hour"].values
    base["metric"] = melted["metric"].values
    base["count_value"] = melted["count_value"].values

    index_cols = [
        col
        for col in [
            "source_file",
            "_merge_key",
            "stop_id",
            "stop_name",
            "district",
            "date",
            "year",
            "month",
            "hour",
            "lat",
            "lon",
            "route_count",
            "route_id",
        ]
        if col in base.columns
    ]
    pivoted = (
        base.pivot_table(index=index_cols, columns="metric", values="count_value", aggfunc="sum")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return pivoted


def standardize_dataframe(df: pd.DataFrame, source_name: str) -> tuple[pd.DataFrame, dict]:
    """원본 데이터프레임 하나를 분석용 표준 데이터프레임으로 변환합니다."""
    cleaned = clean_column_names(df)
    mapping = map_columns(cleaned.columns)
    wide = melt_wide_time_dataframe(cleaned, mapping, source_name)
    if not wide.empty:
        standardized = wide
    else:
        standardized = standardize_long_dataframe(cleaned, mapping, source_name)
    standardized["source_file"] = source_name
    return standardized, mapping


def aggregate_route_data(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """경유 노선 수 또는 노선 ID 데이터를 정류소 단위로 집계합니다."""
    route_frames = [frame for frame in frames if "route_count" in frame.columns or "route_id" in frame.columns]
    if not route_frames:
        return pd.DataFrame()
    route_df = pd.concat(route_frames, ignore_index=True, sort=False)
    if route_df.empty or "_merge_key" not in route_df.columns:
        return pd.DataFrame()

    grouped_parts = []
    if "route_count" in route_df.columns:
        route_count = route_df.groupby("_merge_key", as_index=False)["route_count"].max()
        grouped_parts.append(route_count)
    if "route_id" in route_df.columns:
        route_unique = route_df.dropna(subset=["route_id"]).groupby("_merge_key", as_index=False)["route_id"].nunique()
        route_unique = route_unique.rename(columns={"route_id": "route_count_from_ids"})
        grouped_parts.append(route_unique)

    if not grouped_parts:
        return pd.DataFrame()
    result = grouped_parts[0]
    for part in grouped_parts[1:]:
        result = result.merge(part, on="_merge_key", how="outer")
    if "route_count" not in result.columns:
        result["route_count"] = result["route_count_from_ids"]
    elif "route_count_from_ids" in result.columns:
        result["route_count"] = result["route_count"].fillna(result["route_count_from_ids"])
    return result[["_merge_key", "route_count"]]


def aggregate_route_by_name(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """정류소 ID가 없는 노선 데이터를 정류소명 기준으로 보조 집계합니다."""
    route_frames = [
        frame
        for frame in frames
        if "stop_name" in frame.columns and ("route_count" in frame.columns or "route_id" in frame.columns)
    ]
    if not route_frames:
        return pd.DataFrame()
    route_df = pd.concat(route_frames, ignore_index=True, sort=False)
    if route_df.empty:
        return pd.DataFrame()
    parts = []
    if "route_count" in route_df.columns:
        parts.append(route_df.groupby("stop_name", as_index=False)["route_count"].max())
    if "route_id" in route_df.columns:
        route_ids = route_df.dropna(subset=["route_id"]).groupby("stop_name", as_index=False)["route_id"].nunique()
        route_ids = route_ids.rename(columns={"route_id": "route_count_from_ids"})
        parts.append(route_ids)
    if not parts:
        return pd.DataFrame()
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(part, on="stop_name", how="outer")
    if "route_count" not in result.columns:
        result["route_count"] = result["route_count_from_ids"]
    elif "route_count_from_ids" in result.columns:
        result["route_count"] = result["route_count"].fillna(result["route_count_from_ids"])
    return result[["stop_name", "route_count"]]


def aggregate_location_data(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """위도·경도 데이터를 정류소 단위로 집계합니다."""
    location_frames = [frame for frame in frames if "lat" in frame.columns and "lon" in frame.columns]
    if not location_frames:
        return pd.DataFrame()
    location_df = pd.concat(location_frames, ignore_index=True, sort=False)
    if location_df.empty or "_merge_key" not in location_df.columns:
        return pd.DataFrame()
    result = location_df.groupby("_merge_key", as_index=False).agg({"lat": "median", "lon": "median"})
    return result


def aggregate_location_by_name(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """정류소 ID가 없는 위치 데이터를 정류소명 기준으로 보조 집계합니다."""
    location_frames = [frame for frame in frames if "stop_name" in frame.columns and "lat" in frame.columns and "lon" in frame.columns]
    if not location_frames:
        return pd.DataFrame()
    location_df = pd.concat(location_frames, ignore_index=True, sort=False)
    if location_df.empty:
        return pd.DataFrame()
    agg_map = {"lat": "median", "lon": "median"}
    if "district" in location_df.columns:
        agg_map["district"] = first_non_null
    return location_df.groupby("stop_name", as_index=False).agg(agg_map)


def aggregate_stop_ridership(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """승차·하차·이용객 데이터를 정류소 단위로 합산합니다."""
    ridership_frames = [
        frame
        for frame in frames
        if any(col in frame.columns for col in ["boardings", "alightings", "passengers"])
    ]
    if not ridership_frames:
        return pd.DataFrame()

    ridership = pd.concat(ridership_frames, ignore_index=True, sort=False)
    if ridership.empty or "_merge_key" not in ridership.columns:
        return pd.DataFrame()

    agg_map = {
        "stop_id": first_non_null,
        "stop_name": first_non_null,
        "district": first_non_null,
    }
    for col in ["boardings", "alightings", "passengers"]:
        if col in ridership.columns:
            agg_map[col] = lambda series: series.sum(min_count=1)

    existing_agg = {col: func for col, func in agg_map.items() if col in ridership.columns}
    stop_summary = ridership.groupby("_merge_key", as_index=False).agg(existing_agg)
    return stop_summary


def build_hourly_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """시간대 컬럼이 있는 데이터를 정류소·시간대 단위로 집계합니다."""
    hourly_frames = [
        frame
        for frame in frames
        if "hour" in frame.columns and any(col in frame.columns for col in ["boardings", "alightings", "passengers"])
    ]
    if not hourly_frames:
        return pd.DataFrame()
    hourly = pd.concat(hourly_frames, ignore_index=True, sort=False)
    hourly = hourly[hourly["hour"].notna()].copy()
    if hourly.empty:
        return pd.DataFrame()

    group_cols = [col for col in ["_merge_key", "stop_id", "stop_name", "district", "hour"] if col in hourly.columns]
    agg_cols = {col: lambda series: series.sum(min_count=1) for col in ["boardings", "alightings", "passengers"] if col in hourly.columns}
    return hourly.groupby(group_cols, as_index=False).agg(agg_cols)


def build_monthly_summary(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """연도·월 정보가 있는 데이터를 정류소·월 단위로 집계합니다."""
    monthly_frames = [
        frame
        for frame in frames
        if {"year", "month"}.issubset(frame.columns)
        and any(col in frame.columns for col in ["boardings", "alightings", "passengers"])
    ]
    if not monthly_frames:
        return pd.DataFrame()
    monthly = pd.concat(monthly_frames, ignore_index=True, sort=False)
    monthly = monthly[monthly["year"].notna() & monthly["month"].notna()].copy()
    if monthly.empty:
        return pd.DataFrame()

    group_cols = [col for col in ["_merge_key", "stop_id", "stop_name", "district", "year", "month"] if col in monthly.columns]
    agg_cols = {col: lambda series: series.sum(min_count=1) for col in ["boardings", "alightings", "passengers"] if col in monthly.columns}
    return monthly.groupby(group_cols, as_index=False).agg(agg_cols)


def add_derived_variables(stop_summary: pd.DataFrame, hourly_summary: pd.DataFrame) -> pd.DataFrame:
    """분석에 필요한 전체 이용객, 노선당 승차 인원, 집중도 등을 만듭니다."""
    if stop_summary.empty:
        return stop_summary
    df = stop_summary.copy()

    if "boardings" in df.columns and "alightings" in df.columns:
        df["total_riders"] = df["boardings"].fillna(0) + df["alightings"].fillna(0)
    elif "passengers" in df.columns:
        df["total_riders"] = df["passengers"]

    if "route_count" in df.columns and "boardings" in df.columns:
        df["boardings_per_route"] = safe_divide(df["boardings"], df["route_count"], fill_value=np.nan)

    if not hourly_summary.empty and "hour" in hourly_summary.columns and "boardings" in hourly_summary.columns:
        hourly_for_boarding = hourly_summary.copy()
        hourly_for_boarding["boardings"] = hourly_for_boarding["boardings"].fillna(0)
        morning = (
            hourly_for_boarding[hourly_for_boarding["hour"].between(7, 9)]
            .groupby("_merge_key", as_index=False)["boardings"]
            .sum()
            .rename(columns={"boardings": "morning_boardings"})
        )
        evening = (
            hourly_for_boarding[hourly_for_boarding["hour"].between(17, 20)]
            .groupby("_merge_key", as_index=False)["boardings"]
            .sum()
            .rename(columns={"boardings": "evening_boardings"})
        )
        peak = hourly_for_boarding.loc[
            hourly_for_boarding.groupby("_merge_key")["boardings"].idxmax(), ["_merge_key", "hour"]
        ].copy()
        peak["peak_hour_label"] = peak["hour"].astype(int).astype(str) + "시"
        peak = peak.rename(columns={"hour": "peak_hour"})

        df = df.merge(morning, on="_merge_key", how="left")
        df = df.merge(evening, on="_merge_key", how="left")
        df = df.merge(peak, on="_merge_key", how="left")
        df["morning_concentration"] = safe_divide(df["morning_boardings"], df["boardings"], fill_value=np.nan) * 100
        df["evening_concentration"] = safe_divide(df["evening_boardings"], df["boardings"], fill_value=np.nan) * 100

    if "boardings" in df.columns and "alightings" in df.columns:
        df["boarding_alighting_diff"] = df["boardings"] - df["alightings"]
        denominator = df["boardings"].fillna(0) + df["alightings"].fillna(0)
        df["boarding_alighting_imbalance"] = safe_divide(df["boarding_alighting_diff"].abs(), denominator, fill_value=0)

    return df


def fill_auxiliary_by_stop_name(stop_summary: pd.DataFrame, route_by_name: pd.DataFrame, location_by_name: pd.DataFrame) -> pd.DataFrame:
    """정류소 ID가 서로 맞지 않을 때 정류소명으로 노선·위치 정보를 보완합니다."""
    if stop_summary.empty or "stop_name" not in stop_summary.columns:
        return stop_summary
    result = stop_summary.copy()
    if not route_by_name.empty:
        result = result.merge(route_by_name, on="stop_name", how="left", suffixes=("", "_by_name"))
        if "route_count_by_name" in result.columns:
            result["route_count"] = result.get("route_count", pd.Series(np.nan, index=result.index)).fillna(result["route_count_by_name"])
            result = result.drop(columns=["route_count_by_name"])
    if not location_by_name.empty:
        result = result.merge(location_by_name, on="stop_name", how="left", suffixes=("", "_by_name"))
        for col in ["district", "lat", "lon"]:
            by_name_col = f"{col}_by_name"
            if by_name_col in result.columns:
                result[col] = result.get(col, pd.Series(np.nan, index=result.index)).fillna(result[by_name_col])
                result = result.drop(columns=[by_name_col])
    return result


def prepare_datasets(loaded_files: dict[str, dict]) -> dict:
    """여러 CSV 파일을 읽은 결과를 분석 가능한 하나의 데이터 묶음으로 정리합니다."""
    standardized_frames = []
    hourly_frames = []
    monthly_only_frames = []
    mapping_rows = []
    messages = []

    for file_name, payload in loaded_files.items():
        raw_df = payload["data"]
        standardized, mapping = standardize_dataframe(raw_df, file_name)
        roles = infer_roles(file_name, raw_df.columns, mapping)
        standardized_frames.append(standardized)
        if "hour" in standardized.columns and any(col in standardized.columns for col in ["boardings", "alightings", "passengers"]):
            hourly_frames.append(standardized)
        elif {"year", "month"}.issubset(standardized.columns) and any(
            col in standardized.columns for col in ["boardings", "alightings", "passengers"]
        ):
            monthly_only_frames.append(standardized)
        for standard_col, actual_col in mapping.items():
            mapping_rows.append(
                {
                    "file_name": file_name,
                    "standard_column": standard_col,
                    "actual_column": actual_col,
                    "roles": ", ".join(roles),
                }
            )

    if not standardized_frames:
        return {
            "standardized_rows": pd.DataFrame(),
            "stop_summary": pd.DataFrame(),
            "hourly_summary": pd.DataFrame(),
            "monthly_summary": pd.DataFrame(),
            "column_mapping": pd.DataFrame(mapping_rows),
            "messages": ["data 폴더에 CSV 파일이 없습니다."],
        }

    summary_source_frames = hourly_frames or monthly_only_frames or standardized_frames
    monthly_source_frames = monthly_only_frames or standardized_frames

    stop_summary = aggregate_stop_ridership(summary_source_frames)
    route_summary = aggregate_route_data(standardized_frames)
    route_by_name = aggregate_route_by_name(standardized_frames)
    location_summary = aggregate_location_data(standardized_frames)
    location_by_name = aggregate_location_by_name(standardized_frames)
    hourly_summary = build_hourly_summary(standardized_frames)
    monthly_summary = build_monthly_summary(monthly_source_frames)

    if not route_summary.empty and not stop_summary.empty:
        stop_summary = stop_summary.merge(route_summary, on="_merge_key", how="left")
    if not location_summary.empty and not stop_summary.empty:
        stop_summary = stop_summary.merge(location_summary, on="_merge_key", how="left")

    stop_summary = fill_auxiliary_by_stop_name(stop_summary, route_by_name, location_by_name)
    stop_summary = add_derived_variables(stop_summary, hourly_summary)
    standardized_rows = pd.concat(standardized_frames, ignore_index=True, sort=False)

    return {
        "standardized_rows": standardized_rows,
        "stop_summary": stop_summary,
        "hourly_summary": hourly_summary,
        "monthly_summary": monthly_summary,
        "column_mapping": pd.DataFrame(mapping_rows),
        "messages": messages,
    }
