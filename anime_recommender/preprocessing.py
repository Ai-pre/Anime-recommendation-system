from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from .config import ProjectPaths
from .io import save_pickle


def _safe_feature_name(prefix: str, token: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ["_", "-"] else "_" for ch in str(token).lower())
    safe = safe.strip("_")
    return f"{prefix}_{safe}" if safe else f"{prefix}_unknown"


def make_tfidf_features(series: pd.Series, prefix: str, max_features: int = 100):
    vectorizer = TfidfVectorizer(
        token_pattern=r"[^, ]+",
        lowercase=True,
        max_features=max_features,
    )
    matrix = vectorizer.fit_transform(series.fillna("").astype(str))
    columns = [_safe_feature_name(prefix, name) for name in vectorizer.get_feature_names_out()]
    return pd.DataFrame(matrix.toarray(), columns=columns), vectorizer


def build_meta_preprocessed(paths: ProjectPaths, max_tfidf_features: int = 100) -> pd.DataFrame:
    """Build item-level metadata features for the content-based module."""

    anime = pd.read_csv(paths.anime_csv)
    meta = anime.copy()
    meta = meta.fillna(
        {
            "Genres": "Unknown",
            "Producers": "Unknown",
            "Studios": "Unknown",
            "Licensors": "Unknown",
            "Type": "Unknown",
            "Source": "Unknown",
            "Rating": "Unknown",
            "Premiered": "Unknown",
            "Duration": "Unknown",
        }
    )
    meta["MAL_ID"] = meta["MAL_ID"].astype(int)

    numeric_cols = [
        "Score",
        "Episodes",
        "Ranked",
        "Popularity",
        "Members",
        "Favorites",
        "Watching",
        "Completed",
        "On-Hold",
        "Dropped",
        "Plan to Watch",
        "Score-10",
        "Score-9",
        "Score-8",
        "Score-7",
        "Score-6",
        "Score-5",
        "Score-4",
        "Score-3",
        "Score-2",
        "Score-1",
    ]
    numeric_cols = [col for col in numeric_cols if col in meta.columns]
    for col in numeric_cols:
        meta[col] = pd.to_numeric(meta[col].replace("Unknown", np.nan), errors="coerce")
        meta[col] = meta[col].fillna(meta[col].median())

    scaler = MinMaxScaler()
    scaled_numeric = pd.DataFrame(
        scaler.fit_transform(meta[numeric_cols]),
        columns=numeric_cols,
        index=meta.index,
    )

    label_cols = [col for col in ["Type", "Source", "Rating", "Premiered", "Duration"] if col in meta.columns]
    label_encoders = {}
    encoded = pd.DataFrame(index=meta.index)
    for col in label_cols:
        encoder = LabelEncoder()
        encoded[f"{col}_encoded"] = encoder.fit_transform(meta[col].astype(str))
        label_encoders[col] = encoder

    genre_df, vec_genres = make_tfidf_features(meta["Genres"], "Genre", max_tfidf_features)
    producer_df, vec_producers = make_tfidf_features(meta["Producers"], "Prod", max_tfidf_features)
    studio_df, vec_studios = make_tfidf_features(meta["Studios"], "Studio", max_tfidf_features)

    processed = pd.concat(
        [
            meta[["MAL_ID"]].reset_index(drop=True),
            scaled_numeric.reset_index(drop=True),
            encoded.reset_index(drop=True),
            genre_df.reset_index(drop=True),
            producer_df.reset_index(drop=True),
            studio_df.reset_index(drop=True),
        ],
        axis=1,
    )

    paths.ensure_output_dirs()
    processed.to_csv(paths.meta_preprocessed_csv, index=False)
    save_pickle(
        {
            "label_encoders": label_encoders,
            "vec_genres": vec_genres,
            "vec_producers": vec_producers,
            "vec_studios": vec_studios,
            "scaler": scaler,
            "numeric_cols": numeric_cols,
        },
        paths.encoder_pickle,
    )
    return processed
