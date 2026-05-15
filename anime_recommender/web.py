from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, abort, jsonify, render_template, request, send_file, url_for

from .config import ProjectPaths


DEFAULT_ALLOWED_TYPES = {"TV", "Movie", "OVA", "ONA", "Special"}
BLOCKED_RATINGS = {"Rx - Hentai"}
SERIES_STOP_WORDS = {
    "movie",
    "film",
    "ova",
    "ona",
    "special",
    "season",
    "part",
    "episode",
    "episodes",
    "recap",
    "tv",
    "the",
    "a",
    "an",
    "final",
    "remake",
    "version",
    "edition",
    "i",
    "ii",
    "iii",
    "iv",
    "v",
}


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
        self.catalog["series_key"] = self.catalog["Name"].map(self._series_key)
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
        avoid_same_series: bool = True,
    ) -> tuple[list[dict], list[dict]]:
        allowed_types = allowed_types or DEFAULT_ALLOWED_TYPES
        selected_ids = [int(anime_id) for anime_id in anime_ids if int(anime_id) in self.id_to_idx]
        if not selected_ids:
            raise ValueError("No selected anime were found in the similarity matrix.")

        selected_rows = [self.catalog_by_id.loc[anime_id] for anime_id in selected_ids]
        selected_series = {str(row["series_key"]) for row in selected_rows if str(row["series_key"])}
        selected_idx = [self.id_to_idx[anime_id] for anime_id in selected_ids]
        sims = np.asarray(self.similarity[selected_idx], dtype=np.float32)
        scores = sims.mean(axis=0)

        candidate = self.catalog[self.catalog["MAL_ID"].isin(self.meta_ids)].copy()
        candidate["taste_score"] = candidate["MAL_ID"].map(lambda anime_id: float(scores[self.id_to_idx[int(anime_id)]]))
        candidate = candidate[~candidate["MAL_ID"].isin(selected_ids)]
        candidate = candidate[candidate["Type"].isin(allowed_types)]
        candidate = candidate[~candidate["Rating"].isin(BLOCKED_RATINGS)]
        candidate = candidate[candidate["Members_num"] >= min_members]
        if avoid_same_series and selected_series:
            candidate = candidate[~candidate["series_key"].isin(selected_series)]
            for series_key in selected_series:
                if len(series_key) < 3:
                    continue
                pattern = r"\b" + r"\s+".join(re.escape(token) for token in series_key.split()) + r"\b"
                contains_selected_series = candidate["Name"].str.contains(pattern, case=False, regex=True, na=False)
                candidate = candidate[~contains_selected_series]
        candidate = candidate.sort_values(["taste_score", "Members_num"], ascending=[False, False])
        candidate = self._take_diverse_rows(candidate, top_n, enabled=avoid_same_series)

        selected = [self._anime_summary(row) for row in selected_rows]
        recommendations = [self._anime_summary(row, include_taste=True) for _, row in candidate.iterrows()]
        return selected, recommendations

    @staticmethod
    def _series_key(title: str) -> str:
        """Approximate franchise key for result diversity, not a canonical title parser."""
        text = str(title).lower()
        text = re.split(r"[:(\[\-]", text, maxsplit=1)[0]
        text = re.sub(r"[^a-z0-9]+", " ", text)
        tokens = []
        for token in text.split():
            if token in SERIES_STOP_WORDS:
                continue
            if re.fullmatch(r"\d+(st|nd|rd|th)?", token):
                continue
            tokens.append(token)
        return " ".join(tokens[:4]).strip()

    @staticmethod
    def _take_diverse_rows(candidate: pd.DataFrame, top_n: int, enabled: bool) -> pd.DataFrame:
        if not enabled:
            return candidate.head(top_n)

        kept_indices = []
        seen_series = set()
        for index, row in candidate.iterrows():
            series_key = str(row.get("series_key", ""))
            if series_key and series_key in seen_series:
                continue
            kept_indices.append(index)
            if series_key:
                seen_series.add(series_key)
            if len(kept_indices) >= top_n:
                break
        return candidate.loc[kept_indices]

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
    map_path = paths.artifacts_dir / "anime_tsne_3d.html"

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/map")
    def anime_map():
        if not map_path.exists():
            return render_template("map_missing.html"), 404
        return send_file(map_path)

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
