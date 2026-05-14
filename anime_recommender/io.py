from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def load_pickle(path: Path) -> Any:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def file_size_mb(path: Path) -> float:
    return Path(path).stat().st_size / (1024 * 1024)


def is_lfs_pointer(path: Path) -> bool:
    path = Path(path)
    if not path.exists() or path.stat().st_size > 1024:
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return text.startswith("version https://git-lfs.github.com/spec/v1")


def unwrap_model(package: Any) -> Any:
    if isinstance(package, dict) and "model" in package:
        return package["model"]
    return package


def extract_similarity_matrix(package: Any) -> Any | None:
    if package is None:
        return None
    if not isinstance(package, dict):
        return package
    for key in (
        "item_sim_matrix",
        "item_similarity_matrix",
        "item_sim_matrix_all",
        "similarity_matrix",
        "matrix",
    ):
        if key in package:
            return package[key]
    return None
