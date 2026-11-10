"""Extract calendar-date filters from free-text photo searches."""

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

MONTHS = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
MONTH_PATTERN = "|".join(sorted(MONTHS, key=len, reverse=True))
PERIOD_PATTERN = rf"(?:\d{{1,2}}/\d{{1,2}}/\d{{4}}|(?:{MONTH_PATTERN})\.?\s+\d{{4}}|\d{{4}})"
DATE_QUERY_PATTERN = re.compile(
    rf"(?P<last>\blast\s+year\b)"
    rf"|(?P<in>\bin\s+(?P<in_period>{PERIOD_PATTERN})\b)"
    rf"|(?P<operator>\b(?P<op>before|after)\s+(?P<op_period>{PERIOD_PATTERN})\b)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DateSearch:
    """Date bounds and text left for person-name or visual search."""

    start: date | None
    end: date | None
    remaining_text: str

    @property
    def has_filter(self) -> bool:
        return self.start is not None or self.end is not None


def _period(value: str) -> tuple[date, date]:
    """Return a half-open date interval for a year, month, or US numeric date."""
    if "/" in value:
        try:
            start = datetime.strptime(value, "%m/%d/%Y").date()
        except ValueError as exc:
            raise ValueError(f"Invalid date '{value}'. Use month/day/year, such as 10/01/2013.") from exc
        return start, start + timedelta(days=1)

    pieces = value.removesuffix(".").split()
    if len(pieces) == 2:
        month_name, year_text = pieces
        month = MONTHS[month_name.casefold()]
        year = int(year_text)
        try:
            start = date(year, month, 1)
            end = date(year + (month == 12), month % 12 + 1, 1)
        except ValueError as exc:
            raise ValueError(f"Invalid date period '{value}'.") from exc
        return start, end

    year = int(value)
    try:
        return date(year, 1, 1), date(year + 1, 1, 1)
    except ValueError as exc:
        raise ValueError(f"Invalid year '{value}'.") from exc


def parse_date_search(query: str, *, today: date | None = None) -> DateSearch:
    """Extract supported date clauses, leaving other words untouched for search."""
    today = today or date.today()
    starts: list[date] = []
    ends: list[date] = []
    spans: list[tuple[int, int]] = []

    for match in DATE_QUERY_PATTERN.finditer(query):
        if match.group("last"):
            start, end = date(today.year - 1, 1, 1), date(today.year, 1, 1)
        elif match.group("in"):
            start, end = _period(match.group("in_period"))
        else:
            period_start, period_end = _period(match.group("op_period"))
            if match.group("op").casefold() == "before":
                start, end = None, period_start
            else:
                start, end = period_end, None

        if start is not None:
            starts.append(start)
        if end is not None:
            ends.append(end)
        spans.append(match.span())

    remaining = query
    for start, end in reversed(spans):
        remaining = remaining[:start] + " " + remaining[end:]
    remaining = re.sub(r"\s+", " ", remaining).strip(" \t\r\n,;")

    return DateSearch(
        start=max(starts) if starts else None,
        end=min(ends) if ends else None,
        remaining_text=remaining,
    )
