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


# The recommendation engine ranks movies on TF-IDF over these text fields, so their
# coverage directly bounds how well any given movie can be matched.
TEXT_FIELDS = {
    "overview": "Overview",
    "tagline": "Tagline",
    "keyword_text": "Keywords",
    "cast_text": "Cast",
    "director_text": "Director",
}


def text_coverage(movies: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column, label in TEXT_FIELDS.items():
        present = movies[column].fillna("").str.strip().ne("")
        words = movies.loc[present, column].fillna("").str.split().map(len)
        rows.append(
            {
                "Field": label,
                "Coverage": present.mean(),
                "Mean words": words.mean() if not words.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def movies_per_decade(years: pd.Series) -> pd.Series:
    decades = (years.astype(int) // 10 * 10).astype(int)
    counts = decades.value_counts().sort_index()
    counts.index = counts.index.map(lambda d: f"{d}s")
    return counts


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

    user_rated = movies["user_rating_count"].gt(0)
    print("Text-feature coverage (fields the TF-IDF engine ranks on)")
    coverage = text_coverage(movies)
    coverage_str = coverage.assign(
        Coverage=coverage["Coverage"].map(lambda v: f"{v:.1%}"),
        **{"Mean words": coverage["Mean words"].map(lambda v: f"{v:.1f}")},
    )
    print(coverage_str.to_string(index=False))
    print()

    print(f"MovieLens user-rating coverage: {user_rated.mean():.1%} "
          f"({int(user_rated.sum()):,} of {len(movies):,} movies)")
    print()

    print("Movies per decade")
    print(movies_per_decade(valid_years).to_string())
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
