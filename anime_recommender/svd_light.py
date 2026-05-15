from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LightweightSVD:
    """Small SVD predictor extracted from Surprise SVD factors."""

    global_mean: float
    user_factors: np.ndarray
    item_factors: np.ndarray
    user_bias: np.ndarray
    item_bias: np.ndarray
    raw2inner_user: dict[int, int]
    raw2inner_item: dict[int, int]
    rating_scale: tuple[float, float] = (1.0, 10.0)

    @classmethod
    def from_dict(cls, payload: dict) -> "LightweightSVD":
        return cls(
            global_mean=float(payload["global_mean"]),
            user_factors=payload["user_factors"],
            item_factors=payload["item_factors"],
            user_bias=payload["user_bias"],
            item_bias=payload["item_bias"],
            raw2inner_user={int(k): int(v) for k, v in payload["raw2inner_user"].items()},
            raw2inner_item={int(k): int(v) for k, v in payload["raw2inner_item"].items()},
            rating_scale=tuple(payload.get("rating_scale", (1.0, 10.0))),
        )

    def predict_est(self, user_id: int, anime_id: int, clip: bool = True) -> float:
        user_inner = self.raw2inner_user.get(int(user_id))
        item_inner = self.raw2inner_item.get(int(anime_id))

        est = self.global_mean
        if user_inner is not None:
            est += float(self.user_bias[user_inner])
        if item_inner is not None:
            est += float(self.item_bias[item_inner])
        if user_inner is not None and item_inner is not None:
            est += float(np.dot(self.user_factors[user_inner], self.item_factors[item_inner]))

        if clip:
            low, high = self.rating_scale
            est = float(np.clip(est, low, high))
        return float(est)


def extract_lightweight_svd(svd_package) -> dict:
    """Extract the minimum fields needed to reproduce Surprise SVD estimates."""

    svd = svd_package["model"] if isinstance(svd_package, dict) and "model" in svd_package else svd_package
    trainset = svd.trainset
    rating_scale = getattr(trainset, "rating_scale", (1.0, 10.0))
    return {
        "global_mean": float(trainset.global_mean),
        "user_factors": svd.pu.astype(np.float32, copy=False),
        "item_factors": svd.qi.astype(np.float32, copy=False),
        "user_bias": svd.bu.astype(np.float32, copy=False),
        "item_bias": svd.bi.astype(np.float32, copy=False),
        "raw2inner_user": {int(k): int(v) for k, v in trainset._raw2inner_id_users.items()},
        "raw2inner_item": {int(k): int(v) for k, v in trainset._raw2inner_id_items.items()},
        "rating_scale": (float(rating_scale[0]), float(rating_scale[1])),
    }
