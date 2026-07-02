"""IMDb scores from the official IMDb Non-Commercial Datasets.

title.ratings.tsv.gz (https://developer.imdb.com/non-commercial-datasets/)
carries averageRating (1-10) and numVotes for every IMDb title. The file is
downloaded into data/ on every server startup; IMDb's license does not allow
redistribution, so data/ stays gitignored. Information courtesy of IMDb, used
for non-commercial purposes.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

IMDB_RATINGS_URL = "https://datasets.imdbws.com/title.ratings.tsv.gz"
IMDB_RATINGS_FILENAME = "title.ratings.tsv.gz"

log = logging.getLogger(__name__)


def ensure_imdb_ratings(path: Path, timeout: int = 60, refresh: bool = True) -> Path | None:
    """Refresh title.ratings.tsv.gz, falling back to an existing cache if offline."""
    if path.exists() and not refresh:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.part")
    try:
        log.info("downloading %s -> %s", IMDB_RATINGS_URL, path)
        with urllib.request.urlopen(IMDB_RATINGS_URL, timeout=timeout) as response:
            with tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
        tmp.replace(path)
        return path
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        if path.exists():
            log.warning("could not refresh IMDb ratings dataset (%s); using cached copy", exc)
            return path
        log.warning("could not fetch IMDb ratings dataset (%s); IMDb ratings unavailable", exc)
        return None


def load_imdb_ratings(path: Path, needed_ids: set[int]) -> dict[int, tuple[float, int]]:
    """imdbId -> (averageRating, numVotes), restricted to ids in the catalog."""
    if not needed_ids:
        return {}

    ratings: dict[int, tuple[float, int]] = {}
    chunks = pd.read_csv(
        path,
        sep="\t",
        usecols=["tconst", "averageRating", "numVotes"],
        dtype={"tconst": str, "averageRating": "float32", "numVotes": "int32"},
        chunksize=200_000,
    )
    for chunk in chunks:
        ids = chunk["tconst"].str.slice(2).astype("int64")
        mask = ids.isin(needed_ids)
        if not mask.any():
            continue
        for imdb_id, rating, votes in zip(
            ids[mask],
            chunk.loc[mask, "averageRating"],
            chunk.loc[mask, "numVotes"],
            strict=False,
        ):
            ratings[int(imdb_id)] = (float(rating), int(votes))
    return ratings
