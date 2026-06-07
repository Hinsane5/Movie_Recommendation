"""Rebuild the committed catalogue ``data/movies.csv.gz`` from the raw Kaggle dataset.

The raw TMDB/MovieLens CSVs (``movies_metadata.csv``, ``keywords.csv``, ``credits.csv``,
``links_small.csv``, ``ratings_small.csv``) are hundreds of MB and are NOT committed to
git — they live in ``archive/`` locally. This script parses them into the slim,
gzip-compressed catalogue that the app actually ships with (~14 MB, comfortably under
GitHub's 100 MB limit).

Usage:
    python build_dataset.py            # reads archive/, writes data/movies.csv.gz
    python build_dataset.py --archive /path/to/archive --out data/movies.csv.gz

You only need to run this if you change the raw data or the feature extraction. The
committed ``data/movies.csv.gz`` is already up to date for normal use.
"""

from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

import pandas as pd

from engine import normalize_text  # reuse the same normalisation

BASE_DIR = Path(__file__).resolve().parent
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


def _parse_jsonish_list(value: object) -> list[dict]:
    if pd.isna(value) or not str(value).strip():
        return []
    try:
        parsed = literal_eval(str(value))
    except (ValueError, SyntaxError):
        return []
    return parsed if isinstance(parsed, list) else []


def _names(value: object, limit: int | None = None) -> list[str]:
    items = _parse_jsonish_list(value)
    names = [str(item.get("name", "")).strip() for item in items if isinstance(item, dict)]
    names = [name for name in names if name]
    return names[:limit] if limit else names


def _directors(value: object) -> list[str]:
    crew = _parse_jsonish_list(value)
    return [
        str(p.get("name", "")).strip()
        for p in crew
        if isinstance(p, dict) and p.get("job") == "Director" and p.get("name")
    ][:2]


def _poster(path: object) -> str:
    if pd.isna(path) or not str(path).strip():
        return ""
    text = str(path).strip()
    if text.startswith("http"):
        return text
    return f"{TMDB_IMAGE_BASE_URL}{text}" if text.startswith("/") else ""


def _user_rating_signal(archive: Path) -> pd.DataFrame:
    links_path = archive / "links_small.csv"
    ratings_path = archive / "ratings_small.csv"
    if not links_path.exists() or not ratings_path.exists():
        return pd.DataFrame(columns=["id", "user_rating_mean", "user_rating_count"])
    links = pd.read_csv(links_path)
    ratings = pd.read_csv(ratings_path, usecols=["movieId", "rating"])
    stats = ratings.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    stats = stats.rename(columns={"mean": "user_rating_mean", "count": "user_rating_count"})
    stats = stats.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    stats = stats.dropna(subset=["tmdbId"]).copy()
    stats["id"] = stats["tmdbId"].astype(int)
    return stats[["id", "user_rating_mean", "user_rating_count"]]


def build(archive: Path, out_path: Path) -> None:
    movies = pd.read_csv(archive / "movies_metadata.csv", low_memory=False)
    movies["id"] = pd.to_numeric(movies["id"], errors="coerce")
    movies = movies.dropna(subset=["id", "title"]).copy()
    movies["id"] = movies["id"].astype(int)
    movies = movies.drop_duplicates("id")

    keywords = pd.read_csv(archive / "keywords.csv")
    credits = pd.read_csv(archive / "credits.csv")
    for frame in (keywords, credits):
        frame["id"] = pd.to_numeric(frame["id"], errors="coerce")
        frame.dropna(subset=["id"], inplace=True)
        frame["id"] = frame["id"].astype(int)

    movies = movies.merge(keywords, on="id", how="left")
    movies = movies.merge(credits, on="id", how="left")
    movies = movies.merge(_user_rating_signal(archive), on="id", how="left")
    movies = movies.drop_duplicates("id")

    movies["overview"] = movies["overview"].fillna("")
    movies["tagline"] = movies["tagline"].fillna("")
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies["Release_Year"] = movies["release_date"].dt.year.fillna(0).astype(int)
    for col in ["popularity", "vote_count", "vote_average", "user_rating_mean", "user_rating_count"]:
        movies[col] = pd.to_numeric(movies[col], errors="coerce").fillna(0)

    movies["genre_names"] = movies["genres"].map(_names)
    movies["keyword_names"] = movies["keywords"].map(lambda v: _names(v, 30))
    movies["cast_names"] = movies["cast"].map(lambda v: _names(v, 8))
    movies["director_names"] = movies["crew"].map(_directors)

    movies["Genre"] = movies["genre_names"].map(lambda n: ", ".join(n))
    movies["Poster_Url"] = movies["poster_path"].map(_poster)
    movies["keyword_text"] = movies["keyword_names"].map(lambda v: " ".join(v))
    movies["keyword_display"] = movies["keyword_names"].map(lambda v: ", ".join(v[:8]))
    movies["cast_text"] = movies["cast_names"].map(lambda v: " ".join(v))
    movies["cast_display"] = movies["cast_names"].map(lambda v: ", ".join(v[:4]))
    movies["director_text"] = movies["director_names"].map(lambda v: " ".join(v))
    movies["director_display"] = movies["director_names"].map(lambda v: ", ".join(v))

    has_content = (
        movies["overview"].str.len()
        + movies["tagline"].str.len()
        + movies["Genre"].str.len()
        + movies["keyword_text"].str.len()
    )
    movies = movies[has_content > 0].copy()

    columns = [
        "id", "title", "overview", "tagline", "Genre", "keyword_text", "keyword_display",
        "cast_text", "cast_display", "director_text", "director_display", "Release_Year",
        "original_language", "Poster_Url", "popularity", "vote_count", "vote_average",
        "user_rating_mean", "user_rating_count",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    movies[columns].to_csv(out_path, index=False, compression="gzip")
    print(f"Wrote {len(movies):,} movies -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=BASE_DIR / "archive")
    parser.add_argument("--out", type=Path, default=BASE_DIR / "data" / "movies.csv.gz")
    args = parser.parse_args()
    build(args.archive, args.out)


if __name__ == "__main__":
    main()
