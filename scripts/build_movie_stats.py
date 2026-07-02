"""Aggregate MovieLens ratings.csv into a small per-movie stats cache.

The web app needs per-movie popularity (rating count) and average rating for
confidence signals and display, but ratings.csv is ~877 MB. This script reads
it once in chunks and writes data/movie_stats.csv (~1 MB), which the server
loads at startup.

Usage: .venv/bin/python scripts/build_movie_stats.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RATINGS = ROOT / "ml-32m" / "ratings.csv"
OUT = ROOT / "data" / "movie_stats.csv"


def main() -> None:
    assert RATINGS.exists(), f"missing {RATINGS} — download ml-32m first"
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    chunks = pd.read_csv(
        RATINGS,
        usecols=["movieId", "rating"],
        dtype={"movieId": "int32", "rating": "float32"},
        chunksize=4_000_000,
    )
    for chunk in chunks:
        grouped = chunk.groupby("movieId", sort=False)["rating"].agg(["sum", "count"])
        for movie_id, row in grouped.iterrows():
            movie_id = int(movie_id)
            sums[movie_id] = sums.get(movie_id, 0.0) + float(row["sum"])
            counts[movie_id] = counts.get(movie_id, 0) + int(row["count"])

    movie_ids = np.array(sorted(counts), dtype=np.int64)
    stats = pd.DataFrame(
        {
            "movieId": movie_ids,
            "n_ratings": [counts[m] for m in movie_ids],
            "mean_rating": [round(sums[m] / counts[m], 4) for m in movie_ids],
        }
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(stats):,} movies)")


if __name__ == "__main__":
    main()
