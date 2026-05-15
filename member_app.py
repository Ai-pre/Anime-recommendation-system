from __future__ import annotations

import argparse

from anime_recommender.member_web import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the member-based anime recommender")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8100, type=int)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
