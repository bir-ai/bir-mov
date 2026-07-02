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

RERANK_POOL_MIN = 500
RERANK_POOL_MAX = 4_000
RERANK_POOL_MULTIPLIER = 40
SIMILARITY_PROFILE_LIMIT = 300
MMR_POOL_MULTIPLIER = 10
MMR_POOL_MIN = 200
DIVERSITY_PENALTY = 0.08
DISLIKE_CUTOFF_FALLBACK = 2.5

RERANK_WEIGHTS = {
    "als": 0.50,
    "mf": 0.22,
    "genre": 0.10,
    "quality": 0.08,
    "liked_similarity": 0.08,
    "novelty": 0.02,
    "disliked_similarity": 0.16,
}


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
    match_pct: int          # combined rerank score, normalized within this result set
    als_score: float
    rerank_score: float
    because_of: list[str]   # titles from the user's own liked list
    confidence: str         # high | medium | low


@dataclass
class RerankedCandidate:
    idx: int
    movie: Movie
    als_score: float
    rerank_score: float


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
            "reranker": {
                "pool_min": RERANK_POOL_MIN,
                "pool_max": RERANK_POOL_MAX,
                "pool_multiplier": RERANK_POOL_MULTIPLIER,
                "weights": RERANK_WEIGHTS,
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
        limit = min(top_n, self.n_movies - len(rated_idx))
        if limit <= 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float64)
        data = np.full(len(liked_idx), ALS_ALPHA, dtype=np.float32)
        user_row = sp.csr_matrix(
            (data, (np.zeros(len(liked_idx), dtype=np.int64), liked_idx)),
            shape=(1, self.n_movies),
        )
        ids, scores = self.als.recommend(
            0,
            user_row,
            N=limit,
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

    # ---- reranking helpers ------------------------------------------------

    def _candidate_pool_size(self, count: int, rated_count: int) -> int:
        available = max(0, self.n_movies - rated_count)
        target = max(RERANK_POOL_MIN, count * RERANK_POOL_MULTIPLIER)
        return min(available, RERANK_POOL_MAX, target)

    @staticmethod
    def _minmax(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values.astype(np.float64)
        lo = float(np.min(values))
        hi = float(np.max(values))
        if hi <= lo:
            return np.ones_like(values, dtype=np.float64)
        return ((values - lo) / (hi - lo)).astype(np.float64)

    @staticmethod
    def _ratio_to_top(values: np.ndarray) -> np.ndarray:
        if len(values) == 0:
            return values.astype(np.float64)
        top = float(np.max(values))
        if top <= 0:
            return Recommender._minmax(values)
        return np.clip(values / top, 0.0, 1.0).astype(np.float64)

    @staticmethod
    def _imdb_quality(movie: Movie) -> float:
        rating = 5.8 if movie.imdb_rating is None else movie.imdb_rating
        rating_score = np.clip((rating - 5.5) / 3.0, 0.0, 1.0)
        vote_score = np.clip(np.log1p(movie.imdb_votes) / np.log1p(100_000), 0.0, 1.0)
        return float(0.75 * rating_score + 0.25 * vote_score)

    @staticmethod
    def _novelty(movie: Movie) -> float:
        popularity = np.clip(np.log1p(movie.imdb_votes) / np.log1p(250_000), 0.0, 1.0)
        return float(1.0 - popularity)

    @staticmethod
    def _dislike_cutoff(like_threshold: float) -> float:
        return min(DISLIKE_CUTOFF_FALLBACK, like_threshold - 1.0)

    @staticmethod
    def _genre_profiles(
        known_unique: list[tuple[Movie, float]], like_threshold: float
    ) -> tuple[dict[str, float], dict[str, float]]:
        positive: dict[str, float] = {}
        negative: dict[str, float] = {}
        dislike_cutoff = Recommender._dislike_cutoff(like_threshold)
        for movie, rating in known_unique:
            if rating >= like_threshold:
                weight = max(0.25, rating - like_threshold + 0.5)
                target = positive
            elif rating <= dislike_cutoff:
                weight = max(0.25, dislike_cutoff - rating + 0.5)
                target = negative
            else:
                continue
            for genre in movie.genres:
                target[genre] = target.get(genre, 0.0) + weight
        return positive, negative

    @staticmethod
    def _genre_score(movie: Movie, positive: dict[str, float], negative: dict[str, float]) -> float:
        if not movie.genres:
            return 0.5
        pos_total = sum(positive.values())
        neg_total = sum(negative.values())
        pos_overlap = (
            sum(positive.get(genre, 0.0) for genre in movie.genres) / pos_total
            if pos_total > 0
            else 0.0
        )
        neg_overlap = (
            sum(negative.get(genre, 0.0) for genre in movie.genres) / neg_total
            if neg_total > 0
            else 0.0
        )
        score = 0.5 + 0.5 * np.sqrt(pos_overlap) - 0.5 * np.sqrt(neg_overlap)
        return float(np.clip(score, 0.0, 1.0))

    @staticmethod
    def _profile_sample(
        idx: np.ndarray, ratings5: np.ndarray, mask: np.ndarray, high_first: bool
    ) -> np.ndarray:
        profile_idx = idx[mask]
        if len(profile_idx) <= SIMILARITY_PROFILE_LIMIT:
            return profile_idx
        profile_ratings = ratings5[mask]
        order = np.argsort(-profile_ratings if high_first else profile_ratings)
        return profile_idx[order[:SIMILARITY_PROFILE_LIMIT]]

    def _profile_similarity(self, candidate_idx: np.ndarray, profile_idx: np.ndarray) -> np.ndarray:
        if len(candidate_idx) == 0 or len(profile_idx) == 0:
            return np.zeros(len(candidate_idx), dtype=np.float64)
        sims = self._als_items_unit[candidate_idx] @ self._als_items_unit[profile_idx].T
        return np.clip(np.max(sims, axis=1), 0.0, 1.0).astype(np.float64)

    def _diverse_order(
        self, candidate_idx: np.ndarray, scores: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(candidate_idx) <= count:
            order = np.argsort(-scores)
            return order, scores[order]

        pool_size = min(len(candidate_idx), max(MMR_POOL_MIN, count * MMR_POOL_MULTIPLIER))
        remaining = np.argsort(-scores)[:pool_size]
        selected: list[int] = []
        selected_scores: list[float] = []

        while len(selected) < count and len(remaining) > 0:
            if not selected:
                pick_at = 0
                adjusted_scores = scores[remaining]
            else:
                sims = (
                    self._als_items_unit[candidate_idx[remaining]]
                    @ self._als_items_unit[candidate_idx[np.array(selected, dtype=np.int64)]].T
                )
                duplicate_penalty = np.clip(np.max(sims, axis=1), 0.0, 1.0) * DIVERSITY_PENALTY
                adjusted_scores = scores[remaining] - duplicate_penalty
                pick_at = int(np.argmax(adjusted_scores))
            selected.append(int(remaining[pick_at]))
            selected_scores.append(float(adjusted_scores[pick_at]))
            remaining = np.delete(remaining, pick_at)

        return np.array(selected, dtype=np.int64), np.array(selected_scores, dtype=np.float64)

    def _rerank_candidates(
        self,
        ids: np.ndarray,
        als_scores: np.ndarray,
        mf_scores: np.ndarray,
        catalog: Catalog,
        min_votes: int,
        liked_for_similarity: np.ndarray,
        disliked_for_similarity: np.ndarray,
        genre_positive: dict[str, float],
        genre_negative: dict[str, float],
        count: int,
    ) -> list[RerankedCandidate]:
        candidates: list[tuple[int, Movie, float]] = []
        for rec_i, score in zip(ids, als_scores):
            movie = catalog.movies.get(int(self.movie_ids[rec_i]))
            if movie is None or movie.imdb_votes < min_votes:
                continue
            candidates.append((int(rec_i), movie, float(score)))

        if not candidates:
            return []

        candidate_idx = np.array([idx for idx, _, _ in candidates], dtype=np.int64)
        candidate_als = np.array([score for _, _, score in candidates], dtype=np.float64)
        candidate_mf = mf_scores[candidate_idx]

        als_component = self._ratio_to_top(candidate_als)
        mf_component = (
            0.65 * self._minmax(candidate_mf)
            + 0.35 * np.clip((candidate_mf - 2.5) / 2.5, 0.0, 1.0)
        )
        genre_component = np.array(
            [
                self._genre_score(movie, genre_positive, genre_negative)
                for _, movie, _ in candidates
            ],
            dtype=np.float64,
        )
        quality_component = np.array(
            [self._imdb_quality(movie) for _, movie, _ in candidates], dtype=np.float64
        )
        novelty_component = np.array(
            [self._novelty(movie) for _, movie, _ in candidates], dtype=np.float64
        )
        liked_similarity = self._profile_similarity(candidate_idx, liked_for_similarity)
        disliked_similarity = self._profile_similarity(candidate_idx, disliked_for_similarity)

        rerank_scores = (
            RERANK_WEIGHTS["als"] * als_component
            + RERANK_WEIGHTS["mf"] * mf_component
            + RERANK_WEIGHTS["genre"] * genre_component
            + RERANK_WEIGHTS["quality"] * quality_component
            + RERANK_WEIGHTS["liked_similarity"] * liked_similarity
            + RERANK_WEIGHTS["novelty"] * novelty_component
            - RERANK_WEIGHTS["disliked_similarity"] * disliked_similarity
        )

        order, selected_scores = self._diverse_order(candidate_idx, rerank_scores, count)
        return [
            RerankedCandidate(
                idx=int(candidate_idx[pos]),
                movie=candidates[pos][1],
                als_score=float(candidate_als[pos]),
                rerank_score=float(score),
            )
            for pos, score in zip(order, selected_scores)
        ]

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
        movie_by_idx = {self.movie_id_to_idx[m.movie_id]: m for m, _ in known}
        known_unique = [(movie_by_idx[int(movie_i)], rating) for movie_i, rating in zip(idx, ratings5)]

        liked_idx, like_threshold = self._liked_indices(idx, ratings5)
        mf_scores = self._mf_predict_all(idx, ratings5)
        pool_size = self._candidate_pool_size(count, len(idx))
        ids, als_scores = self._als_rank(liked_idx, idx, top_n=pool_size)
        dislike_cutoff = self._dislike_cutoff(like_threshold)
        liked_for_similarity = self._profile_sample(
            idx, ratings5, ratings5 >= like_threshold, high_first=True
        )
        disliked_for_similarity = self._profile_sample(
            idx, ratings5, ratings5 <= dislike_cutoff, high_first=False
        )
        genre_positive, genre_negative = self._genre_profiles(known_unique, like_threshold)
        ranked = self._rerank_candidates(
            ids=ids,
            als_scores=als_scores,
            mf_scores=mf_scores,
            catalog=catalog,
            min_votes=min_votes,
            liked_for_similarity=liked_for_similarity,
            disliked_for_similarity=disliked_for_similarity,
            genre_positive=genre_positive,
            genre_negative=genre_negative,
            count=count,
        )

        liked_set = set(liked_idx.tolist())
        titles_by_idx = {
            self.movie_id_to_idx[m.movie_id]: m.title
            for m, r in known
            if self.movie_id_to_idx[m.movie_id] in liked_set
        }

        top_score = max((cand.rerank_score for cand in ranked), default=1.0)
        recs: list[Recommendation] = []
        for cand in ranked:
            match_pct = (
                int(round(100 * max(cand.rerank_score, 0.0) / top_score))
                if top_score > 0
                else 0
            )
            confidence = (
                "high" if match_pct >= 75 and cand.movie.imdb_votes >= 10_000
                else "medium" if match_pct >= 45
                else "low"
            )
            predicted5 = float(mf_scores[cand.idx])
            recs.append(
                Recommendation(
                    movie=cand.movie,
                    predicted5=round(predicted5 * 2) / 2,       # Letterboxd uses half stars
                    predicted10=round(predicted5 * 2, 1),
                    match_pct=match_pct,
                    als_score=cand.als_score,
                    rerank_score=cand.rerank_score,
                    because_of=self._because_of(cand.idx, liked_idx, titles_by_idx),
                    confidence=confidence,
                )
            )

        top_genres = [g for g, _ in sorted(genre_positive.items(), key=lambda kv: -kv[1])[:5]]

        profile = UserProfile(
            n_rated=len(matched),
            n_matched=len(known),
            mean_rating5=round(float(ratings5.mean()), 2),
            n_liked=len(liked_idx),
            like_threshold5=like_threshold,
            top_genres=top_genres,
        )
        return profile, recs
