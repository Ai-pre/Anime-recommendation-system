from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from .config import ProjectPaths


def _top_genre(genres: object) -> str:
    if pd.isna(genres):
        return "Unknown"
    first = str(genres).split(",")[0].strip()
    return first or "Unknown"


def _user_ratings(paths: ProjectPaths, user_id: int) -> pd.DataFrame:
    chunks = pd.read_csv(
        paths.rating_csv,
        usecols=["user_id", "anime_id", "rating"],
        chunksize=1_000_000,
    )
    parts = [chunk[chunk["user_id"] == user_id] for chunk in chunks]
    if not parts:
        return pd.DataFrame(columns=["user_id", "anime_id", "rating"])
    return pd.concat(parts, ignore_index=True)


def build_3d_visualization(
    paths: ProjectPaths,
    output: Path | None = None,
    sample_size: int = 6000,
    random_state: int = 42,
    perplexity: float = 30.0,
    user_id: int | None = None,
    like_threshold: float = 8.0,
) -> Path:
    """Create an interactive 3D t-SNE anime embedding HTML file."""

    try:
        import plotly.express as px
    except ImportError as exc:
        raise ImportError("Install plotly before running visualize-3d.") from exc

    paths.ensure_output_dirs()
    output = output or (paths.artifacts_dir / "anime_tsne_3d.html")
    output = Path(output)
    if not output.is_absolute():
        output = paths.root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(paths.meta_preprocessed_csv).fillna(0.0)
    anime = pd.read_csv(paths.anime_csv)
    merged = meta.merge(
        anime[["MAL_ID", "Name", "Genres", "Type"]],
        on="MAL_ID",
        how="left",
    )

    sample_n = min(sample_size, len(merged))
    sample = merged.sample(sample_n, random_state=random_state).copy()

    user_note = ""
    if user_id is not None:
        ratings = _user_ratings(paths, user_id)
        watched = set(ratings["anime_id"].astype(int).tolist())
        liked = set(ratings.loc[ratings["rating"] >= like_threshold, "anime_id"].astype(int).tolist())
        keep_ids = watched | liked
        missing = merged[merged["MAL_ID"].astype(int).isin(keep_ids - set(sample["MAL_ID"].astype(int)))]
        if not missing.empty:
            sample = pd.concat([sample, missing], ignore_index=True).drop_duplicates("MAL_ID")
        user_note = f" | user_id={user_id}, watched={len(watched)}, liked>={like_threshold:g}={len(liked)}"
    else:
        watched = set()
        liked = set()

    feature_cols = [
        col
        for col in meta.columns
        if col != "MAL_ID" and col in sample.columns and pd.api.types.is_numeric_dtype(sample[col])
    ]
    X = sample[feature_cols].to_numpy(dtype=np.float32, copy=True)
    X = StandardScaler(with_mean=True).fit_transform(X)

    n_components = min(50, X.shape[1], max(2, len(sample) - 1))
    X_pca = PCA(n_components=n_components, random_state=random_state).fit_transform(X)
    tsne_perplexity = min(perplexity, max(5.0, (len(sample) - 1) / 3))
    X_tsne = TSNE(
        n_components=3,
        perplexity=tsne_perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state,
    ).fit_transform(X_pca)

    plot_df = sample[["MAL_ID", "Name", "Genres", "Type", "Score", "Popularity"]].copy()
    plot_df[["x", "y", "z"]] = X_tsne
    plot_df["top_genre"] = plot_df["Genres"].map(_top_genre)
    plot_df["highlight"] = "All anime"
    if user_id is not None:
        ids = plot_df["MAL_ID"].astype(int)
        plot_df.loc[ids.isin(watched), "highlight"] = "Watched by user"
        plot_df.loc[ids.isin(liked), "highlight"] = f"Liked by user (rating >= {like_threshold:g})"

    fig = px.scatter_3d(
        plot_df,
        x="x",
        y="y",
        z="z",
        color="highlight" if user_id is not None else "top_genre",
        symbol="highlight" if user_id is not None else None,
        hover_name="Name",
        hover_data={
            "MAL_ID": True,
            "Genres": True,
            "Type": True,
            "Score": True,
            "Popularity": True,
            "x": False,
            "y": False,
            "z": False,
        },
        title=f"Anime Metadata Embedding: PCA + 3D t-SNE (n={len(plot_df)}){user_note}",
        opacity=0.78,
        height=820,
    )
    fig.update_traces(marker={"size": 4})
    fig.update_layout(
        margin={"l": 0, "r": 0, "t": 60, "b": 0},
        legend_title_text="Group" if user_id is not None else "Top genre",
    )
    fig.write_html(output, include_plotlyjs="cdn", full_html=True)
    return output
