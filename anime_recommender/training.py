from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ProjectPaths
from .io import load_pickle, save_pickle
from .preprocessing import build_meta_preprocessed


def rmse_score(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def precision_recall_at_k(frame: pd.DataFrame, pred_col: str, k: int = 10, like_threshold: float = 8.0):
    precisions, recalls = [], []
    for _, group in frame.groupby("user_id"):
        ranked = group.sort_values(pred_col, ascending=False).head(k)
        relevant = group[group["true_rating"] >= like_threshold]
        if relevant.empty:
            continue
        hits = (ranked["true_rating"] >= like_threshold).sum()
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    return float(np.mean(precisions)), float(np.mean(recalls)), len(precisions)


def train_meta_from_ready(paths: ProjectPaths, test_size: float = 0.2, random_state: int = 42) -> pd.DataFrame:
    """Train available meta learners from data/processed/meta_train_ready.csv."""

    paths.ensure_output_dirs()
    meta_ready = pd.read_csv(paths.meta_train_ready_csv)
    X = meta_ready[["cf_score", "cb_score"]]
    y = meta_ready["true_rating"]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    models = {}
    try:
        import lightgbm as lgb

        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=-1,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="regression",
            random_state=random_state,
        )
    except ImportError:
        print("LightGBM is not installed; skipping.")

    try:
        import xgboost as xgb

        models["XGBoost"] = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=random_state,
        )
    except ImportError:
        print("XGBoost is not installed; skipping.")

    try:
        from catboost import CatBoostRegressor

        models["CatBoost"] = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=8.0,
            bagging_temperature=0.5,
            random_strength=2.0,
            loss_function="RMSE",
            verbose=0,
            random_seed=random_state,
        )
    except ImportError:
        print("CatBoost is not installed; skipping.")

    if not models:
        raise RuntimeError("No meta learner libraries are installed.")

    results = []
    val_frame = meta_ready.loc[X_val.index, ["user_id", "anime_id", "true_rating"]].copy()
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        val_frame[f"pred_{name}"] = np.clip(pred, 1.0, 10.0)
        precision, recall, user_count = precision_recall_at_k(val_frame, f"pred_{name}")
        results.append(
            {
                "model": name,
                "rmse": rmse_score(y_val, pred),
                "precision@10": precision,
                "recall@10": recall,
                "evaluated_users": user_count,
            }
        )

    results_df = pd.DataFrame(results).sort_values("rmse")
    best_name = str(results_df.iloc[0]["model"])
    best_model = models[best_name]
    best_rmse = float(results_df.iloc[0]["rmse"])
    save_pickle(
        {
            "model": best_model,
            "model_type": best_name,
            "rmse": best_rmse,
            "feature_cols": ["cf_score", "cb_score"],
        },
        paths.meta_model_pickle,
    )
    return results_df


def train_svd(paths: ProjectPaths, random_state: int = 42):
    try:
        from surprise import Dataset, Reader, SVD
    except ImportError as exc:
        raise ImportError("Install scikit-surprise before running full SVD training.") from exc

    ratings = pd.read_csv(paths.rating_csv, usecols=["user_id", "anime_id", "rating"])
    ratings = ratings[ratings["rating"] > 0]
    reader = Reader(rating_scale=(ratings["rating"].min(), ratings["rating"].max()))
    data = Dataset.load_from_df(ratings[["user_id", "anime_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    svd = SVD(
        n_factors=100,
        n_epochs=20,
        lr_all=0.005,
        reg_all=0.02,
        random_state=random_state,
        verbose=True,
    )
    svd.fit(trainset)
    save_pickle(
        {
            "model": svd,
            "params": {
                "n_factors": 100,
                "n_epochs": 20,
                "lr_all": 0.005,
                "reg_all": 0.02,
            },
        },
        paths.svd_model_pickle,
    )
    return svd, ratings


def compute_item_similarity(paths: ProjectPaths, meta: pd.DataFrame | None = None):
    if meta is None:
        meta = pd.read_csv(paths.meta_preprocessed_csv).fillna(0.0)
    feature_cols = [col for col in meta.columns if col != "MAL_ID"]
    X = StandardScaler().fit_transform(meta[feature_cols].values)
    item_sim_matrix = cosine_similarity(X).astype(np.float32)
    save_pickle(item_sim_matrix, paths.item_similarity_pickle)
    return item_sim_matrix


def train_full_pipeline(
    paths: ProjectPaths,
    sample_users: int = 3000,
    like_threshold: float = 8.0,
    top_k_neighbors: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Full heavy pipeline: preprocessing, SVD, item similarity, meta data, meta learner."""

    paths.ensure_output_dirs()
    meta = build_meta_preprocessed(paths) if not paths.meta_preprocessed_csv.exists() else pd.read_csv(paths.meta_preprocessed_csv)
    svd, ratings = train_svd(paths, random_state=random_state)
    item_sim_matrix = compute_item_similarity(paths, meta)

    malid_to_idx = {int(anime_id): idx for idx, anime_id in enumerate(meta["MAL_ID"].astype(int).values)}
    idx_to_malid = {idx: anime_id for anime_id, idx in malid_to_idx.items()}

    rng = np.random.default_rng(random_state)
    all_users = ratings["user_id"].unique()
    picked_users = rng.choice(all_users, size=min(sample_users, len(all_users)), replace=False)

    records = []
    for user_id in picked_users:
        user_data = ratings[ratings["user_id"] == user_id]
        liked = user_data[user_data["rating"] >= like_threshold]["anime_id"].values
        liked_idx = [malid_to_idx[int(anime_id)] for anime_id in liked if int(anime_id) in malid_to_idx]
        if not liked_idx:
            continue

        liked_sims = item_sim_matrix[liked_idx]
        mask = np.zeros_like(liked_sims, dtype=bool)
        for row_idx in range(len(liked_idx)):
            top_idx = np.argsort(-liked_sims[row_idx])[:top_k_neighbors]
            mask[row_idx, top_idx] = True
        filtered = np.where(mask, liked_sims, 0.0)
        denom = np.maximum(mask.sum(axis=0), 1e-8)
        cb_vector = filtered.sum(axis=0) / denom

        for row in user_data.itertuples(index=False):
            anime_id = int(row.anime_id)
            if anime_id not in malid_to_idx:
                continue
            cf = float(svd.predict(int(user_id), anime_id).est)
            cb = float(cb_vector[malid_to_idx[anime_id]])
            records.append((int(user_id), anime_id, cf, cb, float(row.rating)))

    meta_train = pd.DataFrame(records, columns=["user_id", "anime_id", "cf_score", "cb_score", "true_rating"])
    meta_train.to_csv(paths.meta_train_ready_csv, index=False)
    results_df = train_meta_from_ready(paths, random_state=random_state)

    meta_package = {
        "model": None,
        "model_type": None,
        "rmse": None,
        "malid_to_idx": malid_to_idx,
        "idx_to_malid": idx_to_malid,
        "feature_cols": [col for col in meta.columns if col != "MAL_ID"],
        "training_params": {
            "like_threshold": like_threshold,
            "top_k_neighbors": top_k_neighbors,
            "sample_users": len(picked_users),
            "total_users": len(all_users),
        },
    }
    trained_meta_package = load_pickle(paths.meta_model_pickle)
    meta_package.update(trained_meta_package)
    save_pickle(meta_package, paths.meta_model_pickle)
    return results_df
