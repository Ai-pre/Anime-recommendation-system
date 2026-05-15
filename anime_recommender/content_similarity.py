from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectPaths


def similar_anime_from_features(paths: ProjectPaths, title: str, top_n: int = 10) -> pd.DataFrame:
    """Find similar anime without loading large model pickle files.

    This computes cosine similarity directly from data/processed/meta_preprocessed.csv.
    It is intentionally used by the CLI's `similar` command to avoid loading multi-GB
    SVD/meta/item-similarity artifacts for a content-only lookup.
    """

    anime = pd.read_csv(paths.anime_csv)
    meta = pd.read_csv(paths.meta_preprocessed_csv).fillna(0.0)
    meta["MAL_ID"] = meta["MAL_ID"].astype(int)

    match = anime[anime["Name"].str.contains(title, case=False, na=False, regex=False)]
    if match.empty:
        raise ValueError(f"No anime title matched: {title}")

    target_id = int(match.iloc[0]["MAL_ID"])
    target_name = str(match.iloc[0]["Name"])
    target_positions = np.flatnonzero(meta["MAL_ID"].to_numpy(dtype=np.int64) == target_id)
    if len(target_positions) == 0:
        raise ValueError(f"{target_name} is not in data/processed/meta_preprocessed.csv.")
    target_idx = int(target_positions[0])

    feature_cols = [col for col in meta.columns if col != "MAL_ID"]
    X = meta[feature_cols].to_numpy(dtype=np.float32, copy=True)

    # Match the training-time similarity behavior without materializing a huge matrix.
    means = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = X.std(axis=0, dtype=np.float64).astype(np.float32)
    stds[stds == 0] = 1.0
    X = (X - means) / stds

    target = X[target_idx]
    norms = np.linalg.norm(X, axis=1)
    target_norm = float(np.linalg.norm(target))
    denom = np.maximum(norms * target_norm, 1e-12)
    similarities = (X @ target) / denom
    similarities[target_idx] = -np.inf

    top_idx = np.argsort(-similarities)[:top_n]
    lookup = anime.set_index("MAL_ID", drop=False)

    rows = []
    for rank, idx in enumerate(top_idx, start=1):
        anime_id = int(meta.iloc[idx]["MAL_ID"])
        anime_row = lookup.loc[anime_id]
        rows.append(
            {
                "rank": rank,
                "anime_id": anime_id,
                "title": anime_row["Name"],
                "similarity": round(float(similarities[idx]), 6),
                "genres": anime_row.get("Genres", ""),
                "matched_title": target_name,
            }
        )
    return pd.DataFrame(rows)
