"""Movie catalog: titles, genres, external links, and matching indexes.

Loads movies.csv + links.csv from the MovieLens dump and IMDb's
title.ratings.tsv.gz cache, then builds the lookup structures the parsers and
recommender need:

- imdb_to_movie: exact match for IMDb exports (links.csv carries imdbId)
- title_index: normalized (title, year) match for Letterboxd exports
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .imdb_data import load_imdb_ratings

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")
ARTICLE_RE = re.compile(r"^(.*), (The|A|An|La|Le|Les|L'|Der|Die|Das|Il|El|Los|Las|Un|Une|Een|De|Het|O|Os|As|Um|Uma)$", re.IGNORECASE)
AKA_RE = re.compile(r"\(a\.k\.a\.\s*([^)]+)\)", re.IGNORECASE)
PAREN_ALT_RE = re.compile(r"\(([^)]+)\)")
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Fold a display title into a matching key: accents, case, articles, punctuation."""
    text = unicodedata.normalize("NFKD", title)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.strip()
    moved = ARTICLE_RE.match(text)
    if moved:
        text = f"{moved.group(2)} {moved.group(1)}"
    text = text.lower()
    if text.startswith("the "):
        text = text[4:]
    elif text.startswith("a "):
        text = text[2:]
    elif text.startswith("an "):
        text = text[3:]
    return NON_ALNUM_RE.sub(" ", text).strip()


def front_article(title: str) -> str:
    """Turn MovieLens's 'Usual Suspects, The' into 'The Usual Suspects' for display."""
    moved = ARTICLE_RE.match(title)
    if not moved:
        return title
    article, rest = moved.group(2), moved.group(1)
    joiner = "" if article.endswith("'") else " "
    return f"{article}{joiner}{rest}"


def split_movielens_title(raw: str) -> tuple[str, int | None, list[str]]:
    """Split 'Shawshank Redemption, The (1994)' into (title, year, alt titles)."""
    raw = raw.strip()
    year = None
    year_match = YEAR_RE.search(raw)
    if year_match:
        year = int(year_match.group(1))
        raw = raw[: year_match.start()].strip()

    alts: list[str] = []
    for aka in AKA_RE.findall(raw):
        alts.append(aka.strip())
    base = AKA_RE.sub("", raw).strip()
    # Remaining parenthesised chunks are usually original-language titles.
    for alt in PAREN_ALT_RE.findall(base):
        alt = alt.strip()
        if alt and not alt.isdigit():
            alts.append(alt)
    primary = PAREN_ALT_RE.sub("", base).strip()
    return primary or raw, year, alts


@dataclass
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: list[str]
    imdb_id: int | None
    tmdb_id: int | None
    imdb_rating: float | None = None
    imdb_votes: int = 0

    @property
    def imdb_url(self) -> str | None:
        if self.imdb_id is None:
            return None
        return f"https://www.imdb.com/title/tt{self.imdb_id:07d}/"

    @property
    def letterboxd_url(self) -> str | None:
        # Letterboxd resolves films by TMDB or IMDb id redirects.
        if self.tmdb_id is not None:
            return f"https://letterboxd.com/tmdb/{self.tmdb_id}/"
        if self.imdb_id is not None:
            return f"https://letterboxd.com/imdb/tt{self.imdb_id:07d}/"
        return None


@dataclass
class Catalog:
    movies: dict[int, Movie]
    imdb_to_movie: dict[int, int]
    title_index: dict[str, list[tuple[int | None, int]]] = field(repr=False)

    def by_imdb(self, imdb_id: int) -> Movie | None:
        movie_id = self.imdb_to_movie.get(imdb_id)
        return self.movies.get(movie_id) if movie_id is not None else None

    def by_title_year(self, title: str, year: int | None) -> Movie | None:
        candidates = self.title_index.get(normalize_title(title))
        if not candidates:
            return None
        if year is not None:
            # Exact year first, then ±1 (release-year definitions differ between sites).
            for tolerance in (0, 1):
                for cand_year, movie_id in candidates:
                    if cand_year is not None and abs(cand_year - year) <= tolerance:
                        return self.movies[movie_id]
        if len(candidates) == 1 or year is None:
            return self.movies[candidates[0][1]]
        return None


def load_catalog(data_root: Path, imdb_ratings_path: Path | None = None) -> Catalog:
    movies_df = pd.read_csv(data_root / "movies.csv")
    links_df = pd.read_csv(data_root / "links.csv", dtype={"movieId": "int64"})

    links = {}
    for row in links_df.itertuples():
        imdb = int(row.imdbId) if pd.notna(row.imdbId) else None
        tmdb = int(row.tmdbId) if pd.notna(row.tmdbId) else None
        links[int(row.movieId)] = (imdb, tmdb)

    imdb_ratings: dict[int, tuple[float, int]] = {}
    if imdb_ratings_path is not None and imdb_ratings_path.exists():
        needed_ids = {imdb for imdb, _ in links.values() if imdb is not None}
        imdb_ratings = load_imdb_ratings(imdb_ratings_path, needed_ids)

    movies: dict[int, Movie] = {}
    imdb_to_movie: dict[int, int] = {}
    title_index: dict[str, list[tuple[int | None, int]]] = {}

    for row in movies_df.itertuples():
        movie_id = int(row.movieId)
        title, year, alts = split_movielens_title(str(row.title))
        genres = [] if row.genres == "(no genres listed)" else str(row.genres).split("|")
        imdb_id, tmdb_id = links.get(movie_id, (None, None))
        imdb_rating, imdb_votes = imdb_ratings.get(imdb_id, (None, 0))
        movie = Movie(
            movie_id, front_article(title), year, genres, imdb_id, tmdb_id, imdb_rating, imdb_votes
        )
        movies[movie_id] = movie
        if imdb_id is not None:
            imdb_to_movie[imdb_id] = movie_id
        for key in {normalize_title(title), *(normalize_title(alt) for alt in alts)}:
            if key:
                title_index.setdefault(key, []).append((year, movie_id))

    # Popular movie wins ambiguous same-key matches without a usable year.
    for candidates in title_index.values():
        candidates.sort(key=lambda pair: -(movies[pair[1]].imdb_votes))

    return Catalog(movies=movies, imdb_to_movie=imdb_to_movie, title_index=title_index)


def as_movie_id_index(movie_ids: np.ndarray) -> dict[int, int]:
    """MovieLens movieId -> model movie_idx mapping from movie_ids.npy."""
    return {int(movie_id): idx for idx, movie_id in enumerate(movie_ids)}
