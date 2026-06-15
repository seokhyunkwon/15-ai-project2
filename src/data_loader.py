from pathlib import Path

import pandas as pd

try:
    import chardet
except ImportError:  # pragma: no cover - chardet이 없어도 기본 인코딩 후보로 동작합니다.
    chardet = None


ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]


def list_csv_files(data_dir: str | Path) -> list[Path]:
    """data 폴더와 하위 폴더에서 CSV 파일 목록을 찾습니다."""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []
    return sorted(data_path.rglob("*.csv"))


def detect_encoding(file_path: str | Path) -> str | None:
    """chardet을 이용해 문자 인코딩을 보조적으로 추정합니다."""
    if chardet is None:
        return None
    raw = Path(file_path).read_bytes()[:200_000]
    result = chardet.detect(raw)
    return result.get("encoding") if result else None


def read_csv_with_encoding(file_path: str | Path, **kwargs) -> tuple[pd.DataFrame, str]:
    """여러 인코딩을 순서대로 시도해 CSV 파일을 읽습니다."""
    file_path = Path(file_path)
    encodings = ENCODING_CANDIDATES.copy()
    detected = detect_encoding(file_path)
    if detected and detected not in encodings:
        encodings.append(detected)

    errors = []
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding, low_memory=False, **kwargs)
            return df, encoding
        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    joined_errors = "\n".join(errors)
    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {file_path}\n{joined_errors}")


def audit_dataframe(df: pd.DataFrame, file_path: str | Path, encoding: str) -> dict:
    """읽어온 데이터프레임의 구조, 결측값, 중복 수를 점검합니다."""
    file_path = Path(file_path)
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "rows": int(df.shape[0]),
        "columns_count": int(df.shape[1]),
        "columns": [str(col) for col in df.columns],
        "head": df.head(5).astype(str).to_dict("records"),
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": {str(col): int(count) for col, count in df.isna().sum().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "encoding": encoding,
        "read_error": "",
    }


def inspect_csv_file(file_path: str | Path) -> dict:
    """CSV 파일 하나를 읽고 데이터 점검 결과를 반환합니다."""
    try:
        df, encoding = read_csv_with_encoding(file_path)
        return audit_dataframe(df, file_path, encoding)
    except Exception as exc:
        file_path = Path(file_path)
        return {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "rows": 0,
            "columns_count": 0,
            "columns": [],
            "head": [],
            "dtypes": {},
            "missing_values": {},
            "duplicate_rows": 0,
            "encoding": "",
            "read_error": str(exc),
        }


def inspect_data_folder(data_dir: str | Path) -> list[dict]:
    """data 폴더의 모든 CSV 파일을 점검합니다."""
    return [inspect_csv_file(path) for path in list_csv_files(data_dir)]


def load_csv_files(data_dir: str | Path) -> dict[str, dict]:
    """data 폴더의 모든 CSV를 읽어 파일명별 딕셔너리로 반환합니다."""
    loaded = {}
    for path in list_csv_files(data_dir):
        df, encoding = read_csv_with_encoding(path)
        loaded[path.name] = {
            "data": df,
            "path": str(path),
            "encoding": encoding,
            "audit": audit_dataframe(df, path, encoding),
        }
    return loaded


def audit_to_dataframe(audit: list[dict]) -> pd.DataFrame:
    """앱과 CSV 저장에 적합하도록 점검 결과를 표 형태로 변환합니다."""
    rows = []
    for item in audit:
        rows.append(
            {
                "파일명": item.get("file_name"),
                "행 수": item.get("rows"),
                "열 수": item.get("columns_count"),
                "컬럼명": ", ".join(item.get("columns", [])),
                "중복 데이터 개수": item.get("duplicate_rows"),
                "문자 인코딩": item.get("encoding"),
                "읽기 오류": item.get("read_error"),
            }
        )
    return pd.DataFrame(rows)
