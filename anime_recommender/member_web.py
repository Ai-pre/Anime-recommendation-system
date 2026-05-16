from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.security import check_password_hash, generate_password_hash

from .config import ProjectPaths
from .io import load_pickle, unwrap_model
from .svd_light import LightweightSVD
from .web import BLOCKED_RATINGS, DEFAULT_ALLOWED_TYPES, ContentTasteRecommender


RATING_THRESHOLD = 10
DEFAULT_TOP_N = 12
DEFAULT_MIN_MEMBERS = 5000


class MemberHybridRecommender(ContentTasteRecommender):
    """Hybrid recommender for newly registered users with local ratings."""

    def __init__(self, paths: ProjectPaths):
        super().__init__(paths)
        self.svd_model = self._load_svd(paths)
        self.meta_model = self._load_meta_model(paths)

    def search(self, query: str, limit: int = 12) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        mask = self.catalog["Name"].str.contains(query, case=False, na=False, regex=False)
        rows = (
            self.catalog.loc[mask & self.catalog["MAL_ID"].isin(self.meta_ids)]
            .sort_values(["Popularity_num", "Members_num"], ascending=[True, False])
            .head(limit)
        )
        return [self._anime_summary(row) for _, row in rows.iterrows()]

    @staticmethod
    def _load_svd(paths: ProjectPaths):
        svd_path = paths.svd_light_pickle if paths.svd_light_pickle.exists() else paths.svd_model_pickle
        if not svd_path.exists():
            raise FileNotFoundError(f"Missing SVD model: {paths.svd_light_pickle}")

        package = load_pickle(svd_path)
        if svd_path == paths.svd_light_pickle:
            return LightweightSVD.from_dict(package)
        return unwrap_model(package)

    @staticmethod
    def _load_meta_model(paths: ProjectPaths):
        meta_path = paths.meta_model_core_pickle if paths.meta_model_core_pickle.exists() else paths.meta_model_pickle
        if not meta_path.exists():
            return None
        return unwrap_model(load_pickle(meta_path))

    def personalized_recommend(self, ratings: list[dict], top_n: int = DEFAULT_TOP_N) -> list[dict]:
        usable_ratings = [
            {"anime_id": int(row["anime_id"]), "rating": float(row["rating"])}
            for row in ratings
            if int(row["anime_id"]) in self.id_to_idx
        ]
        if len(usable_ratings) < RATING_THRESHOLD:
            return []

        rated_ids = {row["anime_id"] for row in usable_ratings}
        selected_series = set()
        for anime_id in rated_ids:
            if anime_id in self.catalog_by_id.index:
                series_key = str(self.catalog_by_id.loc[anime_id]["series_key"])
                if series_key:
                    selected_series.add(series_key)

        cf_scores = self._fold_in_cf_scores(usable_ratings)
        cb_scores = self._profile_cb_scores(usable_ratings)

        candidate = self.catalog[self.catalog["MAL_ID"].isin(self.meta_ids)].copy()
        candidate = candidate[~candidate["MAL_ID"].isin(rated_ids)]
        candidate = candidate[candidate["Type"].isin(DEFAULT_ALLOWED_TYPES)]
        candidate = candidate[~candidate["Rating"].isin(BLOCKED_RATINGS)]
        candidate = candidate[candidate["Members_num"] >= DEFAULT_MIN_MEMBERS]
        if selected_series:
            candidate = candidate[~candidate["series_key"].isin(selected_series)]
            for series_key in selected_series:
                if len(series_key) < 3:
                    continue
                pattern = r"\b" + r"\s+".join(re.escape(token) for token in series_key.split()) + r"\b"
                contains_selected_series = candidate["Name"].str.contains(pattern, case=False, regex=True, na=False)
                candidate = candidate[~contains_selected_series]

        candidate["cf_score"] = candidate["MAL_ID"].map(lambda anime_id: float(cf_scores.get(int(anime_id), 0.0)))
        candidate["cb_score"] = candidate["MAL_ID"].map(lambda anime_id: float(cb_scores[self.id_to_idx[int(anime_id)]]))
        candidate = candidate[candidate["cf_score"] > 0.0]
        candidate["predicted_score"] = self._predict_final_scores(candidate["cf_score"], candidate["cb_score"])
        candidate["taste_score"] = candidate["cb_score"]
        candidate = candidate.sort_values(["predicted_score", "Members_num"], ascending=[False, False])
        candidate = self._take_diverse_rows(candidate, top_n, enabled=True)

        rows = []
        for rank, (_, row) in enumerate(candidate.iterrows(), start=1):
            payload = self._anime_summary(row, include_taste=True)
            payload["rank"] = rank
            payload["predicted_score"] = round(float(row["predicted_score"]), 4)
            payload["cf_score"] = round(float(row["cf_score"]), 4)
            payload["cb_score"] = round(float(row["cb_score"]), 4)
            rows.append(payload)
        return rows

    def _fold_in_cf_scores(self, ratings: list[dict]) -> dict[int, float]:
        if isinstance(self.svd_model, LightweightSVD):
            return self._fold_in_light_svd(ratings)
        return self._known_user_fallback_cf(ratings)

    def _fold_in_light_svd(self, ratings: list[dict]) -> dict[int, float]:
        item_inner = []
        y_values = []
        for row in ratings:
            inner = self.svd_model.raw2inner_item.get(int(row["anime_id"]))
            if inner is None:
                continue
            item_inner.append(inner)
            y_values.append(float(row["rating"]) - self.svd_model.global_mean - float(self.svd_model.item_bias[inner]))

        if not item_inner:
            return {}

        q = self.svd_model.item_factors[np.asarray(item_inner, dtype=int)]
        y = np.asarray(y_values, dtype=np.float32)
        x = np.column_stack([np.ones(len(q), dtype=np.float32), q])
        regularization = np.eye(x.shape[1], dtype=np.float32) * 0.12
        regularization[0, 0] = 0.0
        coef = np.linalg.solve(x.T @ x + regularization, x.T @ y)
        user_bias = float(coef[0])
        user_vector = coef[1:].astype(np.float32)

        raw_by_inner = {inner: raw for raw, inner in self.svd_model.raw2inner_item.items()}
        estimates = (
            self.svd_model.global_mean
            + user_bias
            + self.svd_model.item_bias
            + self.svd_model.item_factors @ user_vector
        )
        low, high = self.svd_model.rating_scale
        estimates = np.clip(estimates, low, high)
        return {int(raw_by_inner[idx]): float(score) for idx, score in enumerate(estimates)}

    def _known_user_fallback_cf(self, ratings: list[dict]) -> dict[int, float]:
        # Full Surprise SVD cannot fold in a new user cheaply here, so use a calibrated
        # item popularity prior until svd_light.pkl is available.
        mean_rating = float(np.mean([row["rating"] for row in ratings]))
        centered = mean_rating - 5.5
        scores = {}
        for row in self.catalog.itertuples(index=False):
            anime_id = int(row.MAL_ID)
            base_score = 6.5 if pd.isna(row.Score_num) else float(row.Score_num)
            score = float(np.clip(base_score + centered * 0.15, 1.0, 10.0))
            scores[anime_id] = score
        return scores

    def _profile_cb_scores(self, ratings: list[dict]) -> np.ndarray:
        indices = []
        weights = []
        for row in ratings:
            anime_id = int(row["anime_id"])
            if anime_id not in self.id_to_idx:
                continue
            indices.append(self.id_to_idx[anime_id])
            weights.append(float(row["rating"]) - 5.5)

        if not indices:
            return np.zeros(len(self.meta_ids), dtype=np.float32)

        weights_array = np.asarray(weights, dtype=np.float32)
        if np.allclose(weights_array, 0):
            weights_array = np.ones_like(weights_array)
        sims = np.asarray(self.similarity[indices], dtype=np.float32)
        scores = weights_array @ sims / max(float(np.abs(weights_array).sum()), 1e-8)
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def _predict_final_scores(self, cf_scores: pd.Series, cb_scores: pd.Series) -> np.ndarray:
        if self.meta_model is not None:
            features = pd.DataFrame({"cf_score": cf_scores.astype(float), "cb_score": cb_scores.astype(float)})
            try:
                pred = self.meta_model.predict(features)
            except Exception:
                pred = self.meta_model.predict(features.values)
            return np.clip(np.asarray(pred, dtype=float), 1.0, 10.0)

        cf_norm = (cf_scores.astype(float).to_numpy() - 1.0) / 9.0
        blend = 0.65 * cf_norm + 0.35 * cb_scores.astype(float).to_numpy()
        return np.clip(1.0 + blend * 9.0, 1.0, 10.0)


def _connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_ratings (
            user_id INTEGER NOT NULL,
            anime_id INTEGER NOT NULL,
            rating REAL NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, anime_id),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """
    )
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_app(root: str | Path = ".") -> Flask:
    paths = ProjectPaths(Path(root))
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.secret_key = os.environ.get("ANIME_MEMBER_SECRET", "dev-member-secret-change-me")
    recommender = MemberHybridRecommender(paths)
    db_path = paths.artifacts_dir / "member_site.sqlite3"
    map_path = paths.artifacts_dir / "anime_tsne_3d.html"

    def current_user_id() -> int | None:
        user_id = session.get("user_id")
        return int(user_id) if user_id is not None else None

    def user_payload(user_id: int) -> dict:
        with _connect_db(db_path) as conn:
            user = conn.execute("SELECT id, username FROM users WHERE id = ?", (user_id,)).fetchone()
            rating_rows = conn.execute("SELECT anime_id FROM user_ratings WHERE user_id = ?", (user_id,)).fetchall()
            rating_count = sum(1 for row in rating_rows if int(row["anime_id"]) in recommender.id_to_idx)
        return {
            "id": int(user["id"]),
            "username": str(user["username"]),
            "rating_count": int(rating_count),
            "threshold": RATING_THRESHOLD,
            "ready": int(rating_count) >= RATING_THRESHOLD,
        }

    def require_user() -> int | None:
        user_id = current_user_id()
        if user_id is None:
            return None
        return user_id

    @app.get("/")
    def index():
        return render_template("member_index.html", threshold=RATING_THRESHOLD)

    @app.get("/map")
    def anime_map():
        if not map_path.exists():
            return render_template("map_missing.html"), 404
        return render_template("map.html")

    @app.get("/map/embed")
    def anime_map_embed():
        if not map_path.exists():
            return render_template("map_missing.html"), 404
        return send_file(map_path)

    @app.get("/api/me")
    def api_me():
        user_id = current_user_id()
        if user_id is None:
            return jsonify({"authenticated": False, "threshold": RATING_THRESHOLD})
        return jsonify({"authenticated": True, "user": user_payload(user_id)})

    @app.post("/api/register")
    def api_register():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        if len(username) < 3 or len(password) < 4:
            return jsonify({"error": "아이디는 3글자 이상, 비밀번호는 4글자 이상이어야 합니다."}), 400

        with _connect_db(db_path) as conn:
            try:
                cursor = conn.execute(
                    "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), _now()),
                )
            except sqlite3.IntegrityError:
                return jsonify({"error": "이미 존재하는 아이디입니다."}), 409
            session["user_id"] = int(cursor.lastrowid)
        return jsonify({"user": user_payload(int(session["user_id"]))})

    @app.post("/api/login")
    def api_login():
        payload = request.get_json(silent=True) or {}
        username = str(payload.get("username", "")).strip()
        password = str(payload.get("password", ""))
        with _connect_db(db_path) as conn:
            user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        if user is None or not check_password_hash(str(user["password_hash"]), password):
            return jsonify({"error": "아이디 또는 비밀번호가 맞지 않습니다."}), 401
        session["user_id"] = int(user["id"])
        return jsonify({"user": user_payload(int(user["id"]))})

    @app.post("/api/logout")
    def api_logout():
        session.clear()
        return jsonify({"ok": True})

    @app.get("/api/search")
    def api_search():
        query = request.args.get("q", "")
        return jsonify({"results": recommender.search(query)})

    @app.post("/api/recommend")
    def api_content_recommend():
        payload = request.get_json(silent=True) or {}
        anime_ids = payload.get("anime_ids") or []
        top_n = int(payload.get("top_n") or DEFAULT_TOP_N)
        min_members = int(payload.get("min_members") or DEFAULT_MIN_MEMBERS)
        avoid_same_series = bool(payload.get("avoid_same_series", True))
        try:
            selected, recommendations = recommender.recommend(
                anime_ids,
                top_n=top_n,
                min_members=min_members,
                avoid_same_series=avoid_same_series,
            )
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"selected": selected, "recommendations": recommendations})

    @app.get("/api/ratings")
    def api_ratings():
        user_id = require_user()
        if user_id is None:
            return jsonify({"error": "로그인이 필요합니다."}), 401
        with _connect_db(db_path) as conn:
            rows = conn.execute(
                """
                SELECT anime_id, rating, updated_at
                FROM user_ratings
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        ratings = []
        for row in rows:
            anime = recommender.catalog_by_id.loc[int(row["anime_id"])]
            payload = recommender._anime_summary(anime)
            payload["user_rating"] = float(row["rating"])
            payload["updated_at"] = row["updated_at"]
            ratings.append(payload)
        return jsonify({"ratings": ratings, "user": user_payload(user_id)})

    @app.post("/api/ratings")
    def api_rate():
        user_id = require_user()
        if user_id is None:
            return jsonify({"error": "로그인이 필요합니다."}), 401
        payload = request.get_json(silent=True) or {}
        anime_id = int(payload.get("anime_id", 0))
        rating = float(payload.get("rating", 0))
        if anime_id not in recommender.catalog_by_id.index:
            return jsonify({"error": "알 수 없는 애니입니다."}), 404
        if anime_id not in recommender.id_to_idx:
            return jsonify({"error": "추천 모델에 포함되지 않은 애니입니다."}), 400
        if rating < 1 or rating > 10:
            return jsonify({"error": "평점은 1점부터 10점까지 가능합니다."}), 400

        with _connect_db(db_path) as conn:
            conn.execute(
                """
                INSERT INTO user_ratings (user_id, anime_id, rating, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id, anime_id)
                DO UPDATE SET rating = excluded.rating, updated_at = excluded.updated_at
                """,
                (user_id, anime_id, rating, _now()),
            )
        return jsonify({"ok": True, "user": user_payload(user_id)})

    @app.delete("/api/ratings/<int:anime_id>")
    def api_delete_rating(anime_id: int):
        user_id = require_user()
        if user_id is None:
            return jsonify({"error": "로그인이 필요합니다."}), 401
        with _connect_db(db_path) as conn:
            conn.execute("DELETE FROM user_ratings WHERE user_id = ? AND anime_id = ?", (user_id, anime_id))
        return jsonify({"ok": True, "user": user_payload(user_id)})

    @app.get("/api/recommendations")
    def api_recommendations():
        user_id = require_user()
        if user_id is None:
            return jsonify({"error": "로그인이 필요합니다."}), 401
        with _connect_db(db_path) as conn:
            rows = conn.execute(
                "SELECT anime_id, rating FROM user_ratings WHERE user_id = ?",
                (user_id,),
            ).fetchall()
        ratings = [
            {"anime_id": int(row["anime_id"]), "rating": float(row["rating"])}
            for row in rows
            if int(row["anime_id"]) in recommender.id_to_idx
        ]
        if len(ratings) < RATING_THRESHOLD:
            return jsonify(
                {
                    "ready": False,
                    "rating_count": len(ratings),
                    "threshold": RATING_THRESHOLD,
                    "recommendations": [],
                }
            )
        recommendations = recommender.personalized_recommend(ratings, top_n=DEFAULT_TOP_N)
        return jsonify(
            {
                "ready": True,
                "rating_count": len(ratings),
                "threshold": RATING_THRESHOLD,
                "recommendations": recommendations,
            }
        )

    return app
