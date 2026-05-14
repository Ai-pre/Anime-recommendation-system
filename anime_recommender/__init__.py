"""Anime recommendation training and inference package."""

from .config import ProjectPaths
from .inference import AnimeRecommender

__all__ = ["AnimeRecommender", "ProjectPaths"]
