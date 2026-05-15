from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized file layout for training and inference."""

    root: Path = Path(".")

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser().resolve())

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def anime_csv(self) -> Path:
        return self.raw_dir / "anime.csv"

    @property
    def anime_test_csv(self) -> Path:
        return self.raw_dir / "anime_test.csv"

    @property
    def rating_csv(self) -> Path:
        return self.raw_dir / "rating_complete.csv"

    @property
    def rating_test_csv(self) -> Path:
        return self.raw_dir / "rating_test.csv"

    @property
    def meta_preprocessed_csv(self) -> Path:
        return self.processed_dir / "meta_preprocessed.csv"

    @property
    def meta_train_ready_csv(self) -> Path:
        return self.processed_dir / "meta_train_ready.csv"

    @property
    def encoder_pickle(self) -> Path:
        return self.artifacts_dir / "tfidf_and_encoders.pkl"

    @property
    def svd_model_pickle(self) -> Path:
        return self.models_dir / "svd_model.pkl"

    @property
    def meta_model_pickle(self) -> Path:
        return self.models_dir / "meta_model.pkl"

    @property
    def meta_model_core_pickle(self) -> Path:
        return self.models_dir / "meta_model_core.pkl"

    @property
    def item_similarity_pickle(self) -> Path:
        return self.models_dir / "item_sim_matrix_all.pkl"

    @property
    def item_similarity_npy(self) -> Path:
        return self.models_dir / "item_sim_matrix.float32.npy"

    @property
    def best_meta_model_pickle(self) -> Path:
        return self.models_dir / "best_meta_model.pkl"

    def ensure_output_dirs(self) -> None:
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
