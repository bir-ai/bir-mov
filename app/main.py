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
from .recommender import Recommender

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "ml-32m"
IMDB_RATINGS_PATH = PROJECT_ROOT / "data" / IMDB_RATINGS_FILENAME
WEB_ROOT = PROJECT_ROOT / "web"

MAX_UPLOAD_BYTES = 30 * 1024 * 1024

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


@app.post("/api/recommend")
async def recommend(
    file: UploadFile = File(...),
    count: int = Form(25),
    min_votes: int = Form(50),
) -> dict:
    if _catalog is None or _recommender is None:
        raise HTTPException(status_code=503, detail="Models are still loading, retry shortly.")
    count = max(1, min(100, count))
    min_votes = max(0, min(10_000_000, min_votes))

    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (30 MB limit).")
    try:
        parsed = parse_upload(file.filename or "upload.csv", content)
    except ParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    matched, unmatched = [], []
    for entry in parsed.entries:
        movie = None
        if entry.imdb_id is not None:
            movie = _catalog.by_imdb(entry.imdb_id)
        if movie is None:
            movie = _catalog.by_title_year(entry.title, entry.year)
        if movie is None:
            unmatched.append(f"{entry.title} ({entry.year})" if entry.year else entry.title)
        else:
            matched.append((movie, entry.rating5))

    if not matched:
        raise HTTPException(
            status_code=422,
            detail="No uploaded titles could be matched to the MovieLens catalog.",
        )

    try:
        profile, recs = _recommender.recommend(
            matched, _catalog, count=count, min_votes=min_votes
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {
        "source": parsed.source,
        "profile": {
            "parsed": len(parsed.entries),
            "matched": profile.n_matched,
            "unmatched": len(unmatched),
            "skipped_rows": len(parsed.skipped),
            "mean_rating5": profile.mean_rating5,
            "mean_rating10": round(profile.mean_rating5 * 2, 1),
            "liked_titles": profile.n_liked,
            "like_threshold5": profile.like_threshold5,
            "top_genres": profile.top_genres,
        },
        "recommendations": [
            {"rank": i + 1, **_movie_payload(rec)} for i, rec in enumerate(recs)
        ],
        "unmatched_sample": unmatched[:25],
        "models": _recommender.meta,
    }


app.mount("/", StaticFiles(directory=WEB_ROOT, html=True), name="web")
