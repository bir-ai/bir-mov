"""Parse IMDb and Letterboxd ratings exports into a common shape.

Supported uploads:
- IMDb "Your Ratings" CSV export (Const/Your Rating columns, tt-ids, 1-10 scale)
- Letterboxd ratings.csv (Name/Year/Rating columns, 0.5-5 star scale)
- Letterboxd full-account export .zip (ratings.csv is picked out automatically)

All ratings are converted to the MovieLens 0.5-5.0 scale.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from dataclasses import dataclass

TT_RE = re.compile(r"tt(\d+)")


class ParseError(ValueError):
    """Upload could not be interpreted as an IMDb or Letterboxd export."""


@dataclass
class RatedTitle:
    title: str
    year: int | None
    rating5: float          # MovieLens scale 0.5-5.0
    raw_rating: float       # as found in the file
    imdb_id: int | None = None
    letterboxd_uri: str | None = None


@dataclass
class ParsedRatings:
    source: str             # "imdb" | "letterboxd"
    entries: list[RatedTitle]
    skipped: list[str]      # rows we could not use (no rating / not a film)


def decode_csv_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ParseError("Could not decode the file as text.")


def _rows(text: str) -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise ParseError("The CSV file has no header row.")
    return [row for row in reader]


def _to_int(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip()
    return int(value) if value.isdigit() else None


def parse_imdb_csv(text: str) -> ParsedRatings:
    rows = _rows(text)
    entries: list[RatedTitle] = []
    skipped: list[str] = []
    for row in rows:
        # IMDb has shipped both "Const" and localized/renamed variants over time.
        const = row.get("Const") or row.get("const") or ""
        rating_raw = row.get("Your Rating") or row.get("your rating") or ""
        title = (row.get("Title") or row.get("Original Title") or "").strip()
        title_type = (row.get("Title Type") or "").strip().lower()
        tt = TT_RE.search(const)
        try:
            rating10 = float(rating_raw)
        except ValueError:
            skipped.append(title or const or "?")
            continue
        if title_type and title_type not in {"movie", "tvmovie", "tv movie", "video", "tvspecial", "tv special"}:
            skipped.append(f"{title} ({title_type})")
            continue
        entries.append(
            RatedTitle(
                title=title,
                year=_to_int(row.get("Year")),
                rating5=max(0.5, min(5.0, rating10 / 2.0)),
                raw_rating=rating10,
                imdb_id=int(tt.group(1)) if tt else None,
            )
        )
    if not entries:
        raise ParseError("No rated movies found in the IMDb export.")
    return ParsedRatings(source="imdb", entries=entries, skipped=skipped)


def parse_letterboxd_csv(text: str) -> ParsedRatings:
    rows = _rows(text)
    entries: list[RatedTitle] = []
    skipped: list[str] = []
    for row in rows:
        title = (row.get("Name") or "").strip()
        rating_raw = (row.get("Rating") or "").strip()
        try:
            stars = float(rating_raw)
        except ValueError:
            skipped.append(title or "?")
            continue
        entries.append(
            RatedTitle(
                title=title,
                year=_to_int(row.get("Year")),
                rating5=max(0.5, min(5.0, stars)),
                raw_rating=stars,
                letterboxd_uri=(row.get("Letterboxd URI") or "").strip() or None,
            )
        )
    if not entries:
        raise ParseError("No rated movies found in the Letterboxd export.")
    return ParsedRatings(source="letterboxd", entries=entries, skipped=skipped)


def sniff_csv(text: str) -> str:
    header_line = text.lstrip().splitlines()[0].lower() if text.strip() else ""
    if "const" in header_line and "your rating" in header_line:
        return "imdb"
    if "letterboxd uri" in header_line or ("name" in header_line and "rating" in header_line):
        return "letterboxd"
    raise ParseError(
        "Unrecognized CSV header. Upload an IMDb ratings export or a Letterboxd ratings.csv."
    )


def parse_upload(filename: str, content: bytes) -> ParsedRatings:
    name = filename.lower()
    if name.endswith(".zip"):
        try:
            archive = zipfile.ZipFile(io.BytesIO(content))
        except zipfile.BadZipFile as exc:
            raise ParseError("The .zip file could not be opened.") from exc
        ratings_member = next(
            (m for m in archive.namelist() if m.split("/")[-1] == "ratings.csv"), None
        )
        if ratings_member is None:
            raise ParseError("No ratings.csv found inside the zip (expected a Letterboxd export).")
        text = decode_csv_bytes(archive.read(ratings_member))
        return parse_letterboxd_csv(text)

    text = decode_csv_bytes(content)
    source = sniff_csv(text)
    if source == "imdb":
        return parse_imdb_csv(text)
    return parse_letterboxd_csv(text)
