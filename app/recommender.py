"""Model runtime: ALS fold-in ranking + explicit-MF fold-in score prediction.

The two artifacts trained in ml-32m-train.ipynb split the work:

- ml32m-als128-v1 (implicit ALS) ranks what a new user is most likely to enjoy.
  The notebook evaluation showed ALS dominating explicit MF at top-N ranking.
- ml32m-mf128-v1 (explicit MF) predicts the star rating the user would give,
  which is what powers the "potential score" column.

Neither model stores factors for a brand-new user, so both are folded in from
the uploaded ratings: ALS via implicit's recalculate_user, MF via a closed-form
ridge solve against the frozen item factors.

Artifacts are read from the local model/ directory when present, otherwise
downloaded from the Hugging Face repos (mskayacioglu/ml32m-mf128-v1 and
mskayacioglu/ml32m-als128-v1) into .hf-cache/.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .catalog import Catalog, Movie, as_movie_id_index

HF_REPOS = {
    "ml32m-mf128-v1": "mskayacioglu/ml32m-mf128-v1",
    "ml32m-als128-v1": "mskayacioglu/ml32m-als128-v1",
}
MF_FILES = ("mf_model.pt", "movie_ids.npy")
ALS_FILES = ("als_model.npz", "movie_ids.npy")

ALS_ALPHA = 40.0            # confidence weight used at training time
MF_BIAS_DAMPING = 10.0      # shrinkage for the folded-in user bias
MF_LAMBDA = 0.02            # per-sample L2 used at training time
LIKE_THRESHOLDS = (4.0, 3.5, 3.0)


def resolve_model_dir(name: str, project_root: Path) -> Path:
    """Prefer the local artifact directory; fall back to the Hugging Face repo."""
    local = project_root / "model" / name
    required = MF_FILES if "mf" in name else ALS_FILES
    if all((local / fname).exists() for fname in required):
        return local
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=HF_REPOS[name],
            cache_dir=project_root / ".hf-cache",
        )
    )


@dataclass
class UserProfile:
    n_rated: int
    n_matched: int
    mean_rating5: float
    n_liked: int
    like_threshold5: float
    top_genres: list[str]


@dataclass
class Recommendation:
    movie: Movie
    predicted5: float       # Letterboxd-style 0.5-5.0
    predicted10: float      # IMDb-style 1-10
    match_pct: int          # ALS rank score, normalized within this result set
    als_score: float
    because_of: list[str]   # titles from the user's own liked list
    confidence: str         # high | medium | low


class Recommender:
    def __init__(self, project_root: Path):
        import torch
        from implicit.cpu.als import AlternatingLeastSquares

        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

        als_dir = resolve_model_dir("ml32m-als128-v1", project_root)
        mf_dir = resolve_model_dir("ml32m-mf128-v1", project_root)

        self.als = AlternatingLeastSquares.load(str(als_dir / "als_model.npz"))
        self.movie_ids = np.load(als_dir / "movie_ids.npy")
        self.movie_id_to_idx = as_movie_id_index(self.movie_ids)
        self.n_movies = len(self.movie_ids)

        checkpoint = torch.load(mf_dir / "mf_model.pt", map_location="cpu", weights_only=True)
        state = checkpoint["model_state_dict"]
        self.mf_Q = state["movie_factors.weight"].numpy().astype(np.float64)
        self.mf_bI = state["movie_bias.weight"].numpy().astype(np.float64).ravel()
        self.mf_mu = float(checkpoint["model_config"]["mu"])
        self.meta = {
            "als": {"name": "ml32m-als128-v1", "factors": int(self.als.factors)},
            "mf": {
                "name": checkpoint.get("model_name", "ml32m-mf128-v1"),
                "factors": int(self.mf_Q.shape[1]),
                "val_rmse": float(checkpoint.get("training", {}).get("best_val_rmse", 0.0)),
            },
        }

        item_norms = np.linalg.norm(self.als.item_factors, axis=1, keepdims=True)
        item_norms[item_norms == 0] = 1.0
        self._als_items_unit = self.als.item_factors / item_norms

    # ---- fold-in helpers -------------------------------------------------

    def _mf_predict_all(self, idx: np.ndarray, ratings5: np.ndarray) -> np.ndarray:
        """Closed-form ridge fold-in of a new user into the explicit MF model."""
        n = len(idx)
        residual_bias = ratings5 - self.mf_mu - self.mf_bI[idx]
        b_u = residual_bias.sum() / (n + MF_BIAS_DAMPING)

        Q_u = self.mf_Q[idx]
        y = ratings5 - self.mf_mu - b_u - self.mf_bI[idx]
        lam = max(1.0, MF_LAMBDA * n)
        A = Q_u.T @ Q_u + lam * np.eye(Q_u.shape[1])
        p_u = np.linalg.solve(A, Q_u.T @ y)

        preds = self.mf_mu + b_u + self.mf_bI + self.mf_Q @ p_u
        return np.clip(preds, 0.5, 5.0)

    def _liked_indices(self, idx: np.ndarray, ratings5: np.ndarray) -> tuple[np.ndarray, float]:
        """Adaptive 'liked' threshold: training used >=4.0, relax for sparse histories."""
        for threshold in LIKE_THRESHOLDS:
            liked = idx[ratings5 >= threshold]
            if len(liked) >= 5:
                return liked, threshold
        for threshold in LIKE_THRESHOLDS:
            liked = idx[ratings5 >= threshold]
            if len(liked) >= 1:
                return liked, threshold
        return idx, 0.5

    def _als_rank(
        self, liked_idx: np.ndarray, rated_idx: np.ndarray, top_n: int
    ) -> tuple[np.ndarray, np.ndarray]:
        data = np.full(len(liked_idx), ALS_ALPHA, dtype=np.float32)
        user_row = sp.csr_matrix(
            (data, (np.zeros(len(liked_idx), dtype=np.int64), liked_idx)),
            shape=(1, self.n_movies),
        )
        ids, scores = self.als.recommend(
            0,
            user_row,
            N=min(top_n, self.n_movies - len(rated_idx)),
            filter_already_liked_items=True,
            filter_items=rated_idx,
            recalculate_user=True,
        )
        return np.asarray(ids), np.asarray(scores)

    def _because_of(
        self, rec_idx: int, liked_idx: np.ndarray, titles_by_idx: dict[int, str], k: int = 2
    ) -> list[str]:
        sims = self._als_items_unit[liked_idx] @ self._als_items_unit[rec_idx]
        order = np.argsort(-sims)[:k]
        return [titles_by_idx[int(liked_idx[i])] for i in order if sims[i] > 0.1]

    # ---- public API ------------------------------------------------------

    def recommend(
        self,
        matched: list[tuple[Movie, float]],
        catalog: Catalog,
        count: int = 25,
        min_votes: int = 50,
    ) -> tuple[UserProfile, list[Recommendation]]:
        known = [(m, r) for m, r in matched if m.movie_id in self.movie_id_to_idx]
        if not known:
            raise ValueError("None of the matched movies exist in the model vocabulary.")

        idx = np.array([self.movie_id_to_idx[m.movie_id] for m, _ in known], dtype=np.int64)
        ratings5 = np.array([r for _, r in known], dtype=np.float64)
        # Collapse duplicate entries (rewatches, both-site overlaps) to the mean.
        order = np.argsort(idx)
        idx, ratings5 = idx[order], ratings5[order]
        unique_idx, starts = np.unique(idx, return_index=True)
        ratings5 = np.array([ratings5[s:e].mean() for s, e in zip(starts, [*starts[1:], len(idx)])])
        idx = unique_idx

        liked_idx, like_threshold = self._liked_indices(idx, ratings5)
        mf_scores = self._mf_predict_all(idx, ratings5)
        ids, als_scores = self._als_rank(liked_idx, idx, top_n=count * 5 + len(idx))

        titles_by_idx = {
            self.movie_id_to_idx[m.movie_id]: m.title
            for m, r in known
            if self.movie_id_to_idx[m.movie_id] in set(liked_idx.tolist())
        }

        top_score = float(als_scores[0]) if len(als_scores) else 1.0
        recs: list[Recommendation] = []
        for rec_i, score in zip(ids, als_scores):
            movie = catalog.movies.get(int(self.movie_ids[rec_i]))
            if movie is None or movie.imdb_votes < min_votes:
                continue
            match_pct = int(round(100 * float(score) / top_score)) if top_score > 0 else 0
            confidence = (
                "high" if match_pct >= 75 and movie.imdb_votes >= 10_000
                else "medium" if match_pct >= 45
                else "low"
            )
            predicted5 = float(mf_scores[rec_i])
            recs.append(
                Recommendation(
                    movie=movie,
                    predicted5=round(predicted5 * 2) / 2,       # Letterboxd uses half stars
                    predicted10=round(predicted5 * 2, 1),
                    match_pct=match_pct,
                    als_score=float(score),
                    because_of=self._because_of(int(rec_i), liked_idx, titles_by_idx),
                    confidence=confidence,
                )
            )
            if len(recs) >= count:
                break

        genre_weights: dict[str, float] = {}
        for movie, rating in known:
            if rating >= like_threshold:
                for genre in movie.genres:
                    genre_weights[genre] = genre_weights.get(genre, 0.0) + rating
        top_genres = [g for g, _ in sorted(genre_weights.items(), key=lambda kv: -kv[1])[:5]]

        profile = UserProfile(
            n_rated=len(matched),
            n_matched=len(known),
            mean_rating5=round(float(ratings5.mean()), 2),
            n_liked=len(liked_idx),
            like_threshold5=like_threshold,
            top_genres=top_genres,
        )
        return profile, recs
