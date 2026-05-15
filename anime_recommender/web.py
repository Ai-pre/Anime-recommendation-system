from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, url_for

from .config import ProjectPaths


DEFAULT_ALLOWED_TYPES = {"TV", "Movie", "OVA", "ONA", "Special"}
BLOCKED_RATINGS = {"Rx - Hentai"}


class ContentTasteRecommender:
    """Fast content-based recommender for public web sharing."""

    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.anime = pd.read_csv(paths.anime_csv)
        self.meta_ids = pd.read_csv(paths.meta_preprocessed_csv, usecols=["MAL_ID"])["MAL_ID"].astype(int).to_numpy()
        self.id_to_idx = {int(anime_id): idx for idx, anime_id in enumerate(self.meta_ids)}

        if paths.item_similarity_npy.exists():
            self.similarity = np.load(paths.item_similarity_npy, mmap_mode="r")
        else:
            raise FileNotFoundError(
                "Missing models/item_sim_matrix.float32.npy. Run `python main.py optimize-models` first."
            )

        self.catalog = self.anime.copy()
        self.catalog["MAL_ID"] = self.catalog["MAL_ID"].astype(int)
        self.catalog["Score_num"] = pd.to_numeric(self.catalog["Score"], errors="coerce")
        self.catalog["Members_num"] = pd.to_numeric(self.catalog["Members"], errors="coerce").fillna(0).astype(int)
        self.catalog["Popularity_num"] = pd.to_numeric(self.catalog["Popularity"], errors="coerce")
        self.catalog["top_genre"] = self.catalog["Genres"].fillna("Unknown").map(lambda value: str(value).split(",")[0].strip())
        self.catalog_by_id = self.catalog.set_index("MAL_ID", drop=False)

    def search(self, query: str, limit: int = 12) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        mask = self.catalog["Name"].str.contains(query, case=False, na=False, regex=False)
        rows = self.catalog.loc[mask].sort_values(["Popularity_num", "Members_num"], ascending=[True, False]).head(limit)
        return [self._anime_summary(row) for _, row in rows.iterrows()]

    def recommend(
        self,
        anime_ids: list[int],
        top_n: int = 12,
        min_members: int = 5000,
        allowed_types: set[str] | None = None,
    ) -> tuple[list[dict], list[dict]]:
        allowed_types = allowed_types or DEFAULT_ALLOWED_TYPES
        selected_ids = [int(anime_id) for anime_id in anime_ids if int(anime_id) in self.id_to_idx]
        if not selected_ids:
            raise ValueError("No selected anime were found in the similarity matrix.")

        selected_idx = [self.id_to_idx[anime_id] for anime_id in selected_ids]
        sims = np.asarray(self.similarity[selected_idx], dtype=np.float32)
        scores = sims.mean(axis=0)

        candidate = self.catalog[self.catalog["MAL_ID"].isin(self.meta_ids)].copy()
        candidate["taste_score"] = candidate["MAL_ID"].map(lambda anime_id: float(scores[self.id_to_idx[int(anime_id)]]))
        candidate = candidate[~candidate["MAL_ID"].isin(selected_ids)]
        candidate = candidate[candidate["Type"].isin(allowed_types)]
        candidate = candidate[~candidate["Rating"].isin(BLOCKED_RATINGS)]
        candidate = candidate[candidate["Members_num"] >= min_members]
        candidate = candidate.sort_values(["taste_score", "Members_num"], ascending=[False, False]).head(top_n)

        selected = [self._anime_summary(self.catalog_by_id.loc[anime_id]) for anime_id in selected_ids]
        recommendations = [self._anime_summary(row, include_taste=True) for _, row in candidate.iterrows()]
        return selected, recommendations

    @staticmethod
    def _anime_summary(row: pd.Series, include_taste: bool = False) -> dict:
        score = row.get("Score_num", row.get("Score"))
        try:
            score_value = None if pd.isna(score) else round(float(score), 2)
        except (TypeError, ValueError):
            score_value = None
        payload = {
            "id": int(row["MAL_ID"]),
            "title": str(row["Name"]),
            "genres": str(row.get("Genres", "Unknown")),
            "type": str(row.get("Type", "Unknown")),
            "score": score_value,
            "members": int(row.get("Members_num", row.get("Members", 0)) or 0),
            "rating": str(row.get("Rating", "Unknown")),
            "top_genre": str(row.get("top_genre", "Unknown")),
        }
        if include_taste:
            payload["taste_score"] = round(float(row["taste_score"]), 4)
        return payload


def _connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS recommendation_pages (
            slug TEXT PRIMARY KEY,
            selected_json TEXT NOT NULL,
            recommendations_json TEXT NOT NULL,
            views INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def create_app(root: str | Path = ".") -> Flask:
    paths = ProjectPaths(Path(root))
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    recommender = ContentTasteRecommender(paths)
    db_path = paths.artifacts_dir / "recommendation_pages.sqlite3"

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/search")
    def api_search():
        query = request.args.get("q", "")
        return jsonify({"results": recommender.search(query)})

    @app.post("/api/recommend")
    def api_recommend():
        payload = request.get_json(silent=True) or {}
        anime_ids = payload.get("anime_ids") or []
        top_n = int(payload.get("top_n") or 12)
        min_members = int(payload.get("min_members") or 5000)
        try:
            selected, recommendations = recommender.recommend(anime_ids, top_n=top_n, min_members=min_members)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400

        slug = uuid.uuid4().hex[:10]
        now = datetime.now(timezone.utc).isoformat()
        with _connect_db(db_path) as conn:
            conn.execute(
                """
                INSERT INTO recommendation_pages (slug, selected_json, recommendations_json, views, created_at)
                VALUES (?, ?, ?, 0, ?)
                """,
                (slug, json.dumps(selected, ensure_ascii=False), json.dumps(recommendations, ensure_ascii=False), now),
            )

        share_url = url_for("result_page", slug=slug, _external=False)
        return jsonify(
            {
                "slug": slug,
                "share_url": share_url,
                "selected": selected,
                "recommendations": recommendations,
            }
        )

    @app.get("/r/<slug>")
    def result_page(slug: str):
        with _connect_db(db_path) as conn:
            row = conn.execute("SELECT * FROM recommendation_pages WHERE slug = ?", (slug,)).fetchone()
            if row is None:
                abort(404)
            conn.execute("UPDATE recommendation_pages SET views = views + 1 WHERE slug = ?", (slug,))
            views = int(row["views"]) + 1
        return render_template(
            "result.html",
            slug=slug,
            selected=json.loads(row["selected_json"]),
            recommendations=json.loads(row["recommendations_json"]),
            views=views,
            created_at=row["created_at"],
        )

    @app.get("/api/result/<slug>")
    def api_result(slug: str):
        with _connect_db(db_path) as conn:
            row = conn.execute("SELECT * FROM recommendation_pages WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return jsonify({"error": "not found"}), 404
        return jsonify(
            {
                "slug": slug,
                "selected": json.loads(row["selected_json"]),
                "recommendations": json.loads(row["recommendations_json"]),
                "views": int(row["views"]),
                "created_at": row["created_at"],
            }
        )

    return app
