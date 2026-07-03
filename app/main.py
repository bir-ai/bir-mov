"""bir·mov API: upload an IMDb/Letterboxd ratings export, get recommendations.

Run with:
    .venv/bin/uvicorn app.main:app --reload --port 8360
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles

from .catalog import Catalog, load_catalog
from .imdb_data import IMDB_RATINGS_FILENAME, ensure_imdb_ratings
from .parsers import ParseError, parse_upload
from .recommender import DEFAULT_MIN_VOTES, RecommendationFilters, Recommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "ml-32m"
IMDB_RATINGS_PATH = PROJECT_ROOT / "data" / IMDB_RATINGS_FILENAME
WEB_ROOT = PROJECT_ROOT / "web"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024
MAX_GROUP_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_GROUP_FILES = 8
MAX_VOTE_FILTER = 10_000_000
MIN_RELEASE_YEAR = 1874
MAX_RELEASE_YEAR = 2100

app = FastAPI(title="bir·mov", version="0.1.0")

_catalog: Catalog | None = None
_recommender: Recommender | None = None


@app.on_event("startup")
def load_runtime() -> None:
    global _catalog, _recommender
    imdb_ratings_path = ensure_imdb_ratings(IMDB_RATINGS_PATH)
    _catalog = load_catalog(DATA_ROOT, imdb_ratings_path)
    _recommender = Recommender(PROJECT_ROOT)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "catalog_movies": len(_catalog.movies) if _catalog else 0,
        "imdb_ratings_loaded": (
            any(m.imdb_votes > 0 for m in _catalog.movies.values()) if _catalog else False
        ),
        "models": _recommender.meta if _recommender else None,
    }


def _movie_payload(rec) -> dict:
    movie = rec.movie
    return {
        "movie_id": movie.movie_id,
        "title": movie.title,
        "year": movie.year,
        "genres": movie.genres,
        "imdb_url": movie.imdb_url,
        "letterboxd_url": movie.letterboxd_url,
        "predicted_imdb10": rec.predicted10,
        "predicted_letterboxd5": rec.predicted5,
        "match_pct": rec.match_pct,
        "confidence": rec.confidence,
        "because_of": rec.because_of,
        "imdb_rating": movie.imdb_rating,
        "imdb_votes": movie.imdb_votes,
    }


def _group_movie_payload(rec) -> dict:
    payload = _movie_payload(rec)
    payload.update(
        {
            "group_score": rec.group_score,
            "weakest_predicted_letterboxd5": rec.weakest_predicted5,
            "weakest_predicted_imdb10": round(rec.weakest_predicted5 * 2, 1),
            "agreement_pct": rec.agreement_pct,
            "member_scores": [
                {
                    "name": member.name,
                    "predicted_letterboxd5": member.predicted5,
                    "predicted_imdb10": member.predicted10,
                }
                for member in rec.member_scores
            ],
        }
    )
    return payload


def _match_entries(parsed, catalog: Catalog) -> tuple[list[tuple], list[str]]:
    matched, unmatched = [], []
    for entry in parsed.entries:
        movie = None
        if entry.imdb_id is not None:
            movie = catalog.by_imdb(entry.imdb_id)
        if movie is None:
            movie = catalog.by_title_year(entry.title, entry.year)
        if movie is None:
            unmatched.append(f"{entry.title} ({entry.year})" if entry.year else entry.title)
        else:
            matched.append((movie, entry.rating5))
    return matched, unmatched


def _profile_payload(profile, parsed_count: int, unmatched_count: int, skipped_rows: int) -> dict:
    return {
        "parsed": parsed_count,
        "matched": profile.n_matched,
        "unmatched": unmatched_count,
        "skipped_rows": skipped_rows,
        "mean_rating5": profile.mean_rating5,
        "mean_rating10": round(profile.mean_rating5 * 2, 1),
        "liked_titles": profile.n_liked,
        "like_threshold5": profile.like_threshold5,
        "top_genres": profile.top_genres,
    }


def _upload_display_name(filename: str | None, counts: dict[str, int]) -> str:
    stem = Path(filename or "ratings").stem.strip() or "ratings"
    counts[stem] = counts.get(stem, 0) + 1
    return stem if counts[stem] == 1 else f"{stem} {counts[stem]}"


def _parse_genres(raw: str | None) -> frozenset[str]:
    return frozenset(part.strip() for part in (raw or "").split(",") if part.strip())


def _clamp_optional_int(value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    return max(low, min(high, value))


def _build_filters(
    min_votes: int,
    max_votes: int | None,
    min_year: int | None,
    max_year: int | None,
    include_genres: str | None,
    exclude_genres: str | None,
    genre_match: str,
    min_imdb_rating: float | None,
) -> RecommendationFilters:
    min_votes = max(0, min(MAX_VOTE_FILTER, min_votes))
    max_votes = (
        None
        if max_votes is None or max_votes <= 0
        else max(0, min(MAX_VOTE_FILTER, max_votes))
    )
    if max_votes is not None and max_votes < min_votes:
        raise HTTPException(
            status_code=422,
            detail="Maximum IMDb votes cannot be lower than minimum IMDb votes.",
        )

    min_year = _clamp_optional_int(min_year, MIN_RELEASE_YEAR, MAX_RELEASE_YEAR)
    max_year = _clamp_optional_int(max_year, MIN_RELEASE_YEAR, MAX_RELEASE_YEAR)
    if min_year is not None and max_year is not None and min_year > max_year:
        raise HTTPException(
            status_code=422,
            detail="Minimum year cannot be later than maximum year.",
        )

    include = _parse_genres(include_genres)
    exclude = _parse_genres(exclude_genres)
    overlap = {genre.casefold() for genre in include} & {genre.casefold() for genre in exclude}
    if overlap:
        raise HTTPException(
            status_code=422,
            detail="A genre cannot be both included and excluded.",
        )

    genre_match = "all" if genre_match == "all" else "any"
    min_imdb_rating = (
        None
        if min_imdb_rating is None or min_imdb_rating <= 0
        else max(0.0, min(10.0, min_imdb_rating))
    )

    return RecommendationFilters(
        min_votes=min_votes,
        max_votes=max_votes,
        min_year=min_year,
        max_year=max_year,
        include_genres=include,
        exclude_genres=exclude,
        genre_match=genre_match,
        min_imdb_rating=min_imdb_rating,
    )


@app.post("/api/recommend")
async def recommend(
    file: UploadFile = File(...),
    count: int = Form(25),
    min_votes: int = Form(DEFAULT_MIN_VOTES),
    max_votes: int | None = Form(None),
    min_year: int | None = Form(None),
    max_year: int | None = Form(None),
    include_genres: str | None = Form(None),
    exclude_genres: str | None = Form(None),
    genre_match: str = Form("any"),
    min_imdb_rating: float | None = Form(None),
) -> dict:
    if _catalog is None or _recommender is None:
        raise HTTPException(status_code=503, detail="Models are still loading, retry shortly.")
    count = max(1, min(100, count))
    filters = _build_filters(
        min_votes=min_votes,
        max_votes=max_votes,
        min_year=min_year,
        max_year=max_year,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        genre_match=genre_match,
        min_imdb_rating=min_imdb_rating,
    )

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (30 MB limit).")
    try:
        parsed = parse_upload(file.filename or "upload.csv", content)
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    matched, unmatched = _match_entries(parsed, _catalog)

    if not matched:
        raise HTTPException(
            status_code=422,
            detail="No uploaded titles could be matched to the MovieLens catalog.",
        )

    try:
        profile, recs = _recommender.recommend(
            matched, _catalog, count=count, filters=filters
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "source": parsed.source,
        "profile": _profile_payload(
            profile,
            parsed_count=len(parsed.entries),
            unmatched_count=len(unmatched),
            skipped_rows=len(parsed.skipped),
        ),
        "recommendations": [
            {"rank": i + 1, **_movie_payload(rec)} for i, rec in enumerate(recs)
        ],
        "unmatched_sample": unmatched[:25],
        "models": _recommender.meta,
    }


@app.post("/api/group-recommend")
async def group_recommend(
    files: list[UploadFile] = File(...),
    count: int = Form(25),
    min_votes: int = Form(DEFAULT_MIN_VOTES),
    max_votes: int | None = Form(None),
    min_year: int | None = Form(None),
    max_year: int | None = Form(None),
    include_genres: str | None = Form(None),
    exclude_genres: str | None = Form(None),
    genre_match: str = Form("any"),
    min_imdb_rating: float | None = Form(None),
) -> dict:
    if _catalog is None or _recommender is None:
        raise HTTPException(status_code=503, detail="Models are still loading, retry shortly.")
    if len(files) < 2:
        raise HTTPException(status_code=422, detail="Upload at least two rating files.")
    if len(files) > MAX_GROUP_FILES:
        raise HTTPException(status_code=422, detail=f"Upload at most {MAX_GROUP_FILES} files.")

    count = max(1, min(100, count))
    filters = _build_filters(
        min_votes=min_votes,
        max_votes=max_votes,
        min_year=min_year,
        max_year=max_year,
        include_genres=include_genres,
        exclude_genres=exclude_genres,
        genre_match=genre_match,
        min_imdb_rating=min_imdb_rating,
    )

    total_bytes = 0
    name_counts: dict[str, int] = {}
    grouped_matched: list[tuple[str, list[tuple]]] = []
    user_inputs: list[dict] = []
    exclude_movie_ids: set[int] = set()

    for file in files:
        content = await file.read()
        total_bytes += len(content)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename or 'A file'} is too large (30 MB limit).",
            )
        if total_bytes > MAX_GROUP_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Uploaded files exceed the 100 MB limit.")

        name = _upload_display_name(file.filename, name_counts)
        try:
            parsed = parse_upload(file.filename or "upload.csv", content)
        except ParseError as exc:
            raise HTTPException(status_code=422, detail=f"{name}: {exc}") from exc

        matched, unmatched = _match_entries(parsed, _catalog)
        if not matched:
            raise HTTPException(
                status_code=422,
                detail=f"{name}: no uploaded titles could be matched to the MovieLens catalog.",
            )
        exclude_movie_ids.update(movie.movie_id for movie, _ in matched)
        grouped_matched.append((name, matched))
        user_inputs.append(
            {
                "name": name,
                "source": parsed.source,
                "parsed": len(parsed.entries),
                "skipped": len(parsed.skipped),
                "unmatched": unmatched,
            }
        )

    try:
        group_profile, member_profiles, recs = _recommender.recommend_group(
            grouped_matched,
            _catalog,
            count=count,
            filters=filters,
            exclude_movie_ids=exclude_movie_ids,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    users = []
    unmatched_sample = []
    for user_input, profile in zip(user_inputs, member_profiles):
        users.append(
            {
                "name": user_input["name"],
                "source": user_input["source"],
                **_profile_payload(
                    profile,
                    parsed_count=user_input["parsed"],
                    unmatched_count=len(user_input["unmatched"]),
                    skipped_rows=user_input["skipped"],
                ),
            }
        )
        unmatched_sample.extend(
            f"{user_input['name']}: {title}" for title in user_input["unmatched"][:10]
        )

    return {
        "source": "group",
        "profile": {
            "user_count": group_profile.n_users,
            "parsed": sum(user["parsed"] for user in users),
            "matched": group_profile.n_matched,
            "unmatched": sum(user["unmatched"] for user in users),
            "skipped_rows": sum(user["skipped_rows"] for user in users),
            "mean_rating5": group_profile.mean_rating5,
            "mean_rating10": round(group_profile.mean_rating5 * 2, 1),
            "top_genres": group_profile.top_genres,
            "users": users,
        },
        "recommendations": [
            {"rank": i + 1, **_group_movie_payload(rec)} for i, rec in enumerate(recs)
        ],
        "unmatched_sample": unmatched_sample[:25],
        "models": _recommender.meta,
    }


app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
