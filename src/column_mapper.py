import re
from typing import Iterable


COLUMN_CANDIDATES = {
    "stop_id": [
        "정류소ID",
        "정류소아이디",
        "정류장ID",
        "정류장아이디",
        "정류소번호",
        "정류장번호",
        "정류소코드",
        "정류장코드",
        "bs_id",
        "bus_stop_id",
        "stop_id",
    ],
    "stop_name": [
        "정류소명",
        "정류장명",
        "버스정류소명",
        "정류장",
        "정류소",
        "stop_name",
        "bus_stop_name",
    ],
    "district": [
        "구군",
        "구·군",
        "군구",
        "시군구",
        "행정구역",
        "행정구",
        "구",
        "district",
        "gu",
    ],
    "boardings": [
        "승차인원",
        "승차",
        "승차승객수",
        "승차객수",
        "승차수",
        "승차건수",
        "승차총계",
        "ride_cnt",
        "boarding",
        "boardings",
        "get_on",
        "on_cnt",
    ],
    "alightings": [
        "하차인원",
        "하차",
        "하차승객수",
        "하차객수",
        "하차수",
        "하차건수",
        "하차총계",
        "alighting",
        "alightings",
        "get_off",
        "off_cnt",
    ],
    "passengers": [
        "이용객",
        "이용객수",
        "총이용객",
        "전체이용객",
        "승하차인원",
        "승하차승객수",
        "수송인원",
        "passenger",
        "passengers",
        "passenger_count",
        "total_riders",
    ],
    "lat": ["위도", "latitude", "lat", "y좌표", "y"],
    "lon": ["경도", "longitude", "lon", "lng", "x좌표", "x"],
    "route_count": [
        "경유노선수",
        "노선수",
        "버스노선수",
        "route_count",
        "routes",
        "노선개수",
    ],
    "route_id": [
        "노선ID",
        "노선아이디",
        "노선번호",
        "노선명",
        "route_id",
        "route_no",
        "route_name",
    ],
    "date": ["일자", "날짜", "년월", "연월", "운행일자", "기준일자", "date", "ymd", "yyyymm"],
    "year": ["연도", "년도", "년", "year"],
    "month": ["월", "월별", "기준월", "month"],
    "hour": ["시간", "시간대", "시각", "hour", "time_slot"],
    "metric_type": ["구분", "승하차구분", "승하차", "승하차구분명", "type", "metric"],
}

BOARDING_HINTS = ("승차", "boarding", "boardings", "geton", "get_on", "on_cnt", "ride")
ALIGHTING_HINTS = ("하차", "alighting", "alightings", "getoff", "get_off", "off_cnt")
PASSENGER_HINTS = ("이용", "승하차", "passenger", "total", "수송")


def normalize_column_name(name: str) -> str:
    """컬럼명을 비교하기 쉽도록 공백과 특수문자를 제거하고 소문자로 바꿉니다."""
    text = str(name).strip().lower()
    text = re.sub(r"[\s_\-./\\()\[\]{}:]+", "", text)
    text = text.replace("·", "")
    return text


def find_best_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """후보 이름 목록과 가장 비슷한 실제 컬럼명을 찾습니다."""
    columns = list(columns)
    normalized_columns = {col: normalize_column_name(col) for col in columns}
    normalized_candidates = [normalize_column_name(candidate) for candidate in candidates]

    for col, normalized in normalized_columns.items():
        if normalized in normalized_candidates:
            return col

    scored: list[tuple[int, str]] = []
    for col, normalized in normalized_columns.items():
        for candidate in normalized_candidates:
            if len(candidate) >= 2 and candidate in normalized:
                scored.append((len(candidate), col))
            elif len(normalized) >= 2 and normalized in candidate:
                scored.append((len(normalized), col))
    if not scored:
        return None
    return sorted(scored, reverse=True)[0][1]


def map_columns(columns: Iterable[str]) -> dict:
    """실제 컬럼명 목록에서 분석에 필요한 표준 컬럼명을 자동 탐지합니다."""
    columns = list(columns)
    mapping = {}
    for standard_name, candidates in COLUMN_CANDIDATES.items():
        mapping[standard_name] = find_best_column(columns, candidates)
    return mapping


def parse_hour(value) -> int | None:
    """시간대 값이나 컬럼명에서 0~23 사이의 시간을 추출합니다."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        hour = int(value)
        return hour if 0 <= hour <= 23 else None

    text = str(value).strip().lower()
    patterns = [
        r"(?<!\d)([01]?\d|2[0-3])\s*시",
        r"hour\s*([01]?\d|2[0-3])",
        r"h\s*([01]?\d|2[0-3])",
        r"^([01]?\d|2[0-3])$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            hour = int(match.group(1))
            return hour if 0 <= hour <= 23 else None
    return None


def infer_time_metric_from_column(column: str, source_name: str = "") -> str:
    """시간대 wide 컬럼이 승차, 하차, 이용객 중 무엇을 의미하는지 추정합니다."""
    combined = normalize_column_name(f"{source_name}_{column}")
    if any(hint in combined for hint in BOARDING_HINTS):
        return "boardings"
    if any(hint in combined for hint in ALIGHTING_HINTS):
        return "alightings"
    if any(hint in combined for hint in PASSENGER_HINTS):
        return "passengers"
    return "passengers"


def find_time_columns(columns: Iterable[str], mapped_columns: Iterable[str] | None = None) -> list[dict]:
    """05시, 06시처럼 시간대가 컬럼으로 펼쳐진 wide 형식 컬럼을 찾습니다."""
    mapped = {col for col in (mapped_columns or []) if col}
    results = []
    for col in columns:
        if col in mapped:
            continue
        hour = parse_hour(col)
        if hour is not None:
            results.append({"column": col, "hour": hour})
    return results


def infer_roles(file_name: str, columns: Iterable[str], mapping: dict) -> list[str]:
    """파일명과 컬럼 구조를 바탕으로 데이터 역할을 간단히 추정합니다."""
    roles = set()
    lowered_name = normalize_column_name(file_name)
    time_columns = find_time_columns(columns, mapping.values())

    if mapping.get("boardings") or mapping.get("alightings") or mapping.get("passengers") or time_columns:
        roles.add("ridership")
    if mapping.get("lat") and mapping.get("lon"):
        roles.add("location")
    if mapping.get("route_count") or mapping.get("route_id") or "노선" in lowered_name or "route" in lowered_name:
        roles.add("route")
    if mapping.get("date") or mapping.get("year") or mapping.get("month") or "월별" in lowered_name:
        roles.add("time_series")
    return sorted(roles)
