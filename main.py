from __future__ import annotations

import argparse
from pathlib import Path

from anime_recommender import AnimeRecommender, ProjectPaths
from anime_recommender.content_similarity import similar_anime_from_features
from anime_recommender.io import file_size_mb, is_lfs_pointer
from anime_recommender.training import train_full_pipeline, train_meta_from_ready


def make_paths(root: str | Path) -> ProjectPaths:
    return ProjectPaths(Path(root))


def print_frame(frame, max_rows: int | None = None) -> None:
    if max_rows is not None:
        frame = frame.head(max_rows)
    print(frame.to_string(index=False))


def cmd_check(args) -> None:
    paths = make_paths(args.root)
    required = [
        paths.anime_csv,
        paths.anime_test_csv,
        paths.rating_csv,
        paths.rating_test_csv,
        paths.meta_preprocessed_csv,
        paths.meta_train_ready_csv,
        paths.encoder_pickle,
    ]
    required_models = [
        paths.svd_model_pickle,
        paths.meta_model_pickle,
    ]
    optional_models = [
        paths.item_similarity_pickle,
    ]

    print(f"Project root: {paths.root}")
    print("\nData/artifact files:")
    for path in required:
        if path.exists():
            pointer = " (LFS pointer, replace or run git lfs pull)" if is_lfs_pointer(path) else ""
            print(f"  OK      {path.relative_to(paths.root)}  {file_size_mb(path):.1f} MB{pointer}")
        else:
            print(f"  MISSING {path.relative_to(paths.root)}")

    print("\nModel files:")
    for path in required_models:
        if path.exists():
            print(f"  OK      {path.relative_to(paths.root)}  {file_size_mb(path):.1f} MB")
        else:
            print(f"  MISSING {path.relative_to(paths.root)}")
    for path in optional_models:
        if path.exists():
            print(f"  OPTIONAL {path.relative_to(paths.root)}  {file_size_mb(path):.1f} MB")
        else:
            print(f"  OPTIONAL {path.relative_to(paths.root)}  not present")


def cmd_recommend_user(args) -> None:
    recommender = AnimeRecommender(make_paths(args.root))
    print("Model info:", recommender.model_info())
    result = recommender.recommend_for_user(
        user_id=args.user_id,
        top_n=args.top_n,
        like_threshold=args.like_threshold,
        top_k_neighbors=args.top_k_neighbors,
    )
    print_frame(result)


def cmd_similar(args) -> None:
    result = similar_anime_from_features(make_paths(args.root), args.title, top_n=args.top_n)
    print_frame(result)


def cmd_meta_similar(args) -> None:
    recommender = AnimeRecommender(make_paths(args.root))
    print("Model info:", recommender.model_info())
    result = recommender.recommend_similar_by_meta(
        title=args.title,
        top_n=args.top_n,
        like_threshold=args.like_threshold,
    )
    print_frame(result)


def cmd_train_meta(args) -> None:
    results = train_meta_from_ready(make_paths(args.root), test_size=args.test_size, random_state=args.random_state)
    print_frame(results)
    print("\nSaved meta model to models/meta_model.pkl")


def cmd_train_full(args) -> None:
    results = train_full_pipeline(
        make_paths(args.root),
        sample_users=args.sample_users,
        like_threshold=args.like_threshold,
        top_k_neighbors=args.top_k_neighbors,
        random_state=args.random_state,
    )
    print_frame(results)
    print("\nSaved full pipeline artifacts to models/ and data/processed/")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anime recommendation training and inference CLI")
    parser.add_argument("--root", default=".", help="Project root. Default: current directory")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Check expected data, artifact, and model paths")
    check.set_defaults(func=cmd_check)

    user = sub.add_parser("recommend-user", help="Recommend anime for a user ID")
    user.add_argument("--user-id", type=int, required=True)
    user.add_argument("--top-n", type=int, default=10)
    user.add_argument("--like-threshold", type=float, default=8.0)
    user.add_argument("--top-k-neighbors", type=int, default=30)
    user.set_defaults(func=cmd_recommend_user)

    similar = sub.add_parser("similar", help="Find content-similar anime by title")
    similar.add_argument("--title", required=True)
    similar.add_argument("--top-n", type=int, default=10)
    similar.set_defaults(func=cmd_similar)

    meta_similar = sub.add_parser("meta-similar", help="Recommend items liked by users who liked a title")
    meta_similar.add_argument("--title", required=True)
    meta_similar.add_argument("--top-n", type=int, default=10)
    meta_similar.add_argument("--like-threshold", type=float, default=8.0)
    meta_similar.set_defaults(func=cmd_meta_similar)

    train_meta = sub.add_parser("train-meta", help="Train meta learner from data/processed/meta_train_ready.csv")
    train_meta.add_argument("--test-size", type=float, default=0.2)
    train_meta.add_argument("--random-state", type=int, default=42)
    train_meta.set_defaults(func=cmd_train_meta)

    train_full = sub.add_parser("train-full", help="Run the full heavy training pipeline")
    train_full.add_argument("--sample-users", type=int, default=3000)
    train_full.add_argument("--like-threshold", type=float, default=8.0)
    train_full.add_argument("--top-k-neighbors", type=int, default=30)
    train_full.add_argument("--random-state", type=int, default=42)
    train_full.set_defaults(func=cmd_train_full)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
