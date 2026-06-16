from pathlib import Path

import pandas as pd

from src.analysis import (
    classify_stop_types,
    correlation_analysis,
    generate_analysis_summary_markdown,
    imbalance_candidates,
)
from src.data_loader import audit_to_dataframe, load_csv_files
from src.preprocessing import prepare_datasets
from src.utils import ensure_directories, save_dataframe_csv
from src.weather_analysis import (
    aggregate_bus_monthly,
    calculate_weather_correlations,
    merge_bus_weather_monthly,
    prepare_weather_bundle,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = BASE_DIR / "outputs" / "processed"

HIGH_CONCENTRATION_THRESHOLD = 25.0
LOW_USAGE_QUANTILE = 0.25


def build_processed_outputs() -> dict:
    """원본 CSV를 읽어 Streamlit 앱이 바로 사용할 정제 CSV를 생성합니다."""
    ensure_directories(BASE_DIR)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    loaded_files = load_csv_files(DATA_DIR)
    audit = [payload.get("audit", {}) for payload in loaded_files.values()]

    bundle = prepare_datasets(loaded_files)
    bundle.update(prepare_weather_bundle(loaded_files))
    bundle["audit"] = audit

    stop_summary = bundle.get("stop_summary", pd.DataFrame())
    if not stop_summary.empty:
        bundle["stop_summary"] = classify_stop_types(
            stop_summary,
            morning_threshold=HIGH_CONCENTRATION_THRESHOLD,
            evening_threshold=HIGH_CONCENTRATION_THRESHOLD,
            low_usage_quantile=LOW_USAGE_QUANTILE,
        )

    candidates, _ = imbalance_candidates(bundle.get("stop_summary", pd.DataFrame()))

    save_dataframe_csv(audit_to_dataframe(audit), PROCESSED_DIR / "data_check_report.csv")
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
        save_dataframe_csv(bundle.get(name, pd.DataFrame()), PROCESSED_DIR / f"{name}.csv")

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

    bus_monthly = aggregate_bus_monthly(bundle.get("monthly_summary", pd.DataFrame()), "전체 대구")
    weather_merged = merge_bus_weather_monthly(bus_monthly, bundle.get("weather_monthly", pd.DataFrame()))
    save_dataframe_csv(weather_merged, PROCESSED_DIR / "bus_weather_monthly_merged.csv")
    save_dataframe_csv(calculate_weather_correlations(weather_merged), PROCESSED_DIR / "weather_correlation_results.csv")

    return {
        "stop_rows": len(bundle.get("stop_summary", pd.DataFrame())),
        "hourly_rows": len(bundle.get("hourly_summary", pd.DataFrame())),
        "monthly_rows": len(bundle.get("monthly_summary", pd.DataFrame())),
        "weather_months": len(bundle.get("weather_monthly", pd.DataFrame())),
        "weather_merged_months": len(weather_merged),
    }


def main() -> None:
    """PowerShell에서 실행할 전처리 진입점입니다."""
    result = build_processed_outputs()
    print("전처리 완료")
    for key, value in result.items():
        print(f"- {key}: {value:,}")


if __name__ == "__main__":
    main()
