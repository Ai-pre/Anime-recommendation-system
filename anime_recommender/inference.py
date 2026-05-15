from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .io import extract_similarity_matrix, load_pickle, unwrap_model


@dataclass
class AnimeRecommender:
    """Load saved models and run recommendation inference."""

    paths: ProjectPaths = ProjectPaths()

    def __post_init__(self) -> None:
        self.paths = ProjectPaths(self.paths.root if isinstance(self.paths, ProjectPaths) else self.paths)
        self.svd_package = None
        self.meta_package = None
        self.svd_model = None
        self.meta_model = None
        self.item_sim_matrix = None
        self.similarity_source = None
        self.malid_to_idx: dict[int, int] = {}
        self.idx_to_malid: dict[int, int] = {}
        self._anime_df: pd.DataFrame | None = None
        self._ratings_df: pd.DataFrame | None = None
        self._meta_ready: pd.DataFrame | None = None
        self._anime_lookup: pd.DataFrame | None = None
        self.load_models()

    @property
    def anime_df(self) -> pd.DataFrame:
        if self._anime_df is None:
            self._anime_df = pd.read_csv(self.paths.anime_csv)
        return self._anime_df

    @property
    def ratings_df(self) -> pd.DataFrame:
        if self._ratings_df is None:
            self._ratings_df = pd.read_csv(self.paths.rating_csv, usecols=["user_id", "anime_id", "rating"])
            self._ratings_df = self._ratings_df[self._ratings_df["rating"] > 0]
        return self._ratings_df

    @property
    def meta_ready(self) -> pd.DataFrame:
        if self._meta_ready is None:
            self._meta_ready = pd.read_csv(self.paths.meta_train_ready_csv)
        return self._meta_ready

    @property
    def anime_lookup(self) -> pd.DataFrame:
        if self._anime_lookup is None:
            self._anime_lookup = self.anime_df.set_index("MAL_ID", drop=False)
        return self._anime_lookup

    def load_models(self) -> None:
        if not self.paths.svd_model_pickle.exists():
            raise FileNotFoundError(f"Missing SVD model: {self.paths.svd_model_pickle}")
        if not self.paths.meta_model_pickle.exists():
            raise FileNotFoundError(f"Missing meta model: {self.paths.meta_model_pickle}")

        self.svd_package = load_pickle(self.paths.svd_model_pickle)
        self.meta_package = load_pickle(self.paths.meta_model_pickle)

        self.svd_model = unwrap_model(self.svd_package)
        self.meta_model = unwrap_model(self.meta_package)

        if isinstance(self.meta_package, dict):
            self.malid_to_idx = {int(k): int(v) for k, v in self.meta_package.get("malid_to_idx", {}).items()}
            self.idx_to_malid = {int(k): int(v) for k, v in self.meta_package.get("idx_to_malid", {}).items()}
            self.item_sim_matrix = extract_similarity_matrix(self.meta_package)
            if self.item_sim_matrix is not None:
                self.similarity_source = "meta_model.pkl"

        if not self.malid_to_idx or not self.idx_to_malid:
            meta_ids = pd.read_csv(self.paths.meta_preprocessed_csv, usecols=["MAL_ID"])["MAL_ID"].astype(int).tolist()
            self.malid_to_idx = {anime_id: idx for idx, anime_id in enumerate(meta_ids)}
            self.idx_to_malid = {idx: anime_id for anime_id, idx in self.malid_to_idx.items()}

        if self.item_sim_matrix is None and self.paths.item_similarity_pickle.exists():
            self.item_sim_matrix = extract_similarity_matrix(load_pickle(self.paths.item_similarity_pickle))
            if self.item_sim_matrix is not None:
                self.similarity_source = "item_sim_matrix_all.pkl"

    def model_info(self) -> dict[str, object]:
        model_type = "Unknown"
        rmse = None
        if isinstance(self.meta_package, dict):
            model_type = self.meta_package.get("model_type", model_type)
            rmse = self.meta_package.get("rmse")
        return {
            "model_type": model_type,
            "rmse": rmse,
            "anime_count": len(self.malid_to_idx),
            "has_item_similarity": self.item_sim_matrix is not None,
            "similarity_source": self.similarity_source,
        }

    def _require_similarity(self) -> None:
        if self.item_sim_matrix is None:
            raise RuntimeError(
                "Item similarity matrix was not found. Use a meta_model.pkl package that contains "
                "item_sim_matrix, or put item_sim_matrix_all.pkl in models/ as a fallback."
            )

    def get_cb_scores_for_user(self, user_id: int, like_threshold: float = 8.0, top_k_neighbors: int = 30):
        self._require_similarity()
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        liked_anime = user_ratings[user_ratings["rating"] >= like_threshold]["anime_id"].values
        liked_idx = [self.malid_to_idx[int(anime_id)] for anime_id in liked_anime if int(anime_id) in self.malid_to_idx]
        if not liked_idx:
            return None

        liked_sims = self.item_sim_matrix[liked_idx]
        mask = np.zeros_like(liked_sims, dtype=bool)
        for row_idx in range(len(liked_idx)):
            top_idx = np.argsort(-liked_sims[row_idx])[:top_k_neighbors]
            mask[row_idx, top_idx] = True

        filtered = np.where(mask, liked_sims, 0.0)
        denom = np.maximum(mask.sum(axis=0), 1e-8)
        return filtered.sum(axis=0) / denom

    def _predict_meta(self, cf_score: float, cb_score: float) -> float:
        features = pd.DataFrame([[cf_score, cb_score]], columns=["cf_score", "cb_score"])
        try:
            pred = self.meta_model.predict(features)
        except Exception:
            pred = self.meta_model.predict(features.values)
        return float(np.clip(pred[0], 1.0, 10.0))

    def recommend_for_user(
        self,
        user_id: int,
        top_n: int = 10,
        like_threshold: float = 8.0,
        top_k_neighbors: int = 30,
    ) -> pd.DataFrame:
        user_history = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_history.empty:
            raise ValueError(f"User {user_id} was not found in rating_complete.csv.")

        cb_vector = self.get_cb_scores_for_user(user_id, like_threshold, top_k_neighbors)
        if cb_vector is None:
            raise ValueError(f"User {user_id} has no anime rated >= {like_threshold}.")

        watched_ids = set(user_history["anime_id"].astype(int).values)
        candidates = [
            int(anime_id)
            for anime_id in self.anime_df["MAL_ID"].astype(int).values
            if int(anime_id) not in watched_ids and int(anime_id) in self.malid_to_idx
        ]

        scores = []
        for anime_id in candidates:
            cf = float(self.svd_model.predict(user_id, anime_id).est)
            cb = float(cb_vector[self.malid_to_idx[anime_id]])
            pred = self._predict_meta(cf, cb)
            scores.append((anime_id, pred, cf, cb))

        scores.sort(key=lambda row: row[1], reverse=True)
        rows = []
        for rank, (anime_id, pred, cf, cb) in enumerate(scores[:top_n], start=1):
            anime = self.anime_lookup.loc[anime_id]
            rows.append(
                {
                    "rank": rank,
                    "anime_id": anime_id,
                    "title": anime["Name"],
                    "predicted_score": round(pred, 4),
                    "cf_score": round(cf, 4),
                    "cb_score": round(cb, 4),
                    "genres": anime.get("Genres", ""),
                }
            )
        return pd.DataFrame(rows)

    def similar_anime(self, title: str, top_n: int = 10) -> pd.DataFrame:
        self._require_similarity()
        match = self.anime_df[self.anime_df["Name"].str.contains(title, case=False, na=False, regex=False)]
        if match.empty:
            raise ValueError(f"No anime title matched: {title}")

        target_id = int(match.iloc[0]["MAL_ID"])
        target_name = str(match.iloc[0]["Name"])
        if target_id not in self.malid_to_idx:
            raise ValueError(f"{target_name} is not in the processed similarity matrix.")

        target_idx = self.malid_to_idx[target_id]
        similarities = self.item_sim_matrix[target_idx]
        top_idx = np.argsort(-similarities)[1 : top_n + 1]

        rows = []
        for rank, idx in enumerate(top_idx, start=1):
            anime_id = int(self.idx_to_malid[int(idx)])
            anime = self.anime_lookup.loc[anime_id]
            rows.append(
                {
                    "rank": rank,
                    "anime_id": anime_id,
                    "title": anime["Name"],
                    "similarity": round(float(similarities[idx]), 6),
                    "genres": anime.get("Genres", ""),
                    "matched_title": target_name,
                }
            )
        return pd.DataFrame(rows)

    def recommend_similar_by_meta(
        self,
        title: str,
        top_n: int = 10,
        like_threshold: float = 8.0,
    ) -> pd.DataFrame:
        match = self.anime_df[self.anime_df["Name"].str.contains(title, case=False, na=False, regex=False)]
        if match.empty:
            raise ValueError(f"No anime title matched: {title}")

        target_id = int(match.iloc[0]["MAL_ID"])
        target_name = str(match.iloc[0]["Name"])

        liked_users = self.ratings_df[
            (self.ratings_df["anime_id"] == target_id) & (self.ratings_df["rating"] >= like_threshold)
        ]["user_id"].unique()
        if len(liked_users) == 0:
            raise ValueError(f"No users rated {target_name} >= {like_threshold}.")

        candidate = self.ratings_df[
            (self.ratings_df["user_id"].isin(liked_users)) & (self.ratings_df["anime_id"] != target_id)
        ][["user_id", "anime_id"]]
        merged = candidate.merge(
            self.meta_ready[["user_id", "anime_id", "cf_score", "cb_score"]],
            on=["user_id", "anime_id"],
            how="inner",
        )
        if merged.empty:
            raise ValueError("No candidate rows had matching meta features.")

        try:
            preds = self.meta_model.predict(merged[["cf_score", "cb_score"]])
        except Exception:
            preds = self.meta_model.predict(merged[["cf_score", "cb_score"]].values)
        merged["predicted_score"] = np.clip(preds, 1.0, 10.0)

        ranked = (
            merged.groupby("anime_id")
            .agg(predicted_score=("predicted_score", "mean"), user_count=("user_id", "nunique"))
            .reset_index()
            .sort_values(["predicted_score", "user_count"], ascending=False)
            .head(top_n)
        )
        ranked = ranked.merge(
            self.anime_df[["MAL_ID", "Name", "Genres"]].rename(
                columns={"MAL_ID": "anime_id", "Name": "title", "Genres": "genres"}
            ),
            on="anime_id",
            how="left",
        )
        ranked.insert(0, "rank", range(1, len(ranked) + 1))
        ranked["matched_title"] = target_name
        ranked["predicted_score"] = ranked["predicted_score"].round(4)
        return ranked[["rank", "anime_id", "title", "predicted_score", "user_count", "genres", "matched_title"]]
