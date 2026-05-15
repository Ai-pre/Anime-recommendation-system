from __future__ import annotations

import numpy as np

from .config import ProjectPaths
from .io import extract_similarity_matrix, load_pickle, save_pickle


def optimize_meta_model(paths: ProjectPaths) -> dict[str, str]:
    """Split the huge meta model package into a lightweight model and mmap matrix.

    The original project stores the meta learner and item similarity matrix together
    in models/meta_model.pkl. That is convenient, but expensive to load with SVD in
    the same Python process. This function keeps the same model logic while making
    inference memory friendlier:

    - models/meta_model_core.pkl: meta learner package without item_sim_matrix
    - models/item_sim_matrix.float32.npy: float32 matrix loaded with mmap_mode='r'
    """

    if not paths.meta_model_pickle.exists():
        raise FileNotFoundError(f"Missing meta model: {paths.meta_model_pickle}")

    package = load_pickle(paths.meta_model_pickle)
    if not isinstance(package, dict):
        raise TypeError("Expected meta_model.pkl to contain a dictionary package.")

    matrix = extract_similarity_matrix(package)
    if matrix is None:
        raise KeyError("meta_model.pkl does not contain item_sim_matrix.")

    paths.ensure_output_dirs()
    np.save(paths.item_similarity_npy, matrix.astype(np.float32, copy=False))

    core_package = dict(package)
    core_package.pop("item_sim_matrix", None)
    save_pickle(core_package, paths.meta_model_core_pickle)

    return {
        "meta_model_core": str(paths.meta_model_core_pickle),
        "item_similarity_npy": str(paths.item_similarity_npy),
    }
