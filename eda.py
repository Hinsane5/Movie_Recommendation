"""Run EDA for the shipped movie recommendation dataset.

This script is intentionally separate from the Streamlit UI. It reads the deployable
artifacts in ``data/`` and prints the same summary saved in ``docs/eda_results.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CATALOGUE_PATH = BASE_DIR / "data" / "movies.csv.gz"
CF_PATH = BASE_DIR / "data" / "cf_neighbors.csv.gz"


def split_genres(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .str.split(", ")
        .explode()
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )


def main() -> None:
    movies = pd.read_csv(CATALOGUE_PATH)
    cf_neighbors = pd.read_csv(CF_PATH) if CF_PATH.exists() else pd.DataFrame()

    valid_years = movies["Release_Year"].replace(0, np.nan).dropna()
    poster_coverage = movies["Poster_Url"].fillna("").ne("").mean()
    cf_covered = cf_neighbors["movie_id"].nunique() if not cf_neighbors.empty else 0

    print("Movie catalogue EDA")
    print("===================")
    print(f"Movies: {len(movies):,}")
    print(f"Columns: {len(movies.columns):,}")
    print(f"Release years: {int(valid_years.min())}-{int(valid_years.max())}")
    print(f"Poster coverage: {poster_coverage:.1%}")
    print(f"Collaborative-filtering covered movies: {cf_covered:,}")
    print(f"Collaborative-filtering coverage rate: {cf_covered / len(movies):.1%}")
    print()

    print("Top genres")
    print(split_genres(movies["Genre"]).value_counts().head(10).to_string())
    print()

    print("Top original languages")
    print(movies["original_language"].fillna("unknown").str.upper().value_counts().head(10).to_string())
    print()

    print("Top-rated movies, minimum 1,000 TMDB votes")
    top_rated = (
        movies[movies["vote_count"] >= 1000]
        .sort_values(["vote_average", "vote_count"], ascending=False)
        [["title", "Release_Year", "Genre", "vote_average", "vote_count"]]
        .head(10)
    )
    print(top_rated.to_string(index=False))


if __name__ == "__main__":
    main()
