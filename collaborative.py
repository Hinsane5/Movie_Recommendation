"""Item-based collaborative filtering ("Because users who liked X also liked Y").

The content engine in ``engine.py`` matches a *free-text description* to movie metadata.
This module adds the complementary signal: a memory-based **item-based collaborative
filtering** model trained on the MovieLens user x movie ratings matrix.

How it works
------------
1. Build a sparse *item x user* ratings matrix from ``ratings.csv`` (movieId mapped to
   the catalogue's TMDB ``id`` via ``links.csv``).
2. Mean-center each rating by the rating user's average (**adjusted cosine** similarity),
   which removes the "some users rate everything high" bias.
3. L2-normalise each item row, then compute item-item cosine similarity and keep only the
   top-K most-similar movies per title.

Like ``data/movies.csv.gz``, the raw ratings CSVs are hundreds of MB and are NOT committed
to git. So this is precomputed **offline** into a slim, gzip-compressed neighbour table
``data/cf_neighbors.csv.gz`` (a few MB) which is what the deployed app actually loads.

Usage
-----
    python collaborative.py                 # build from archive/, write the neighbour table
    python collaborative.py --small         # use ratings_small.csv (fast, less coverage)
    python collaborative.py --top-k 30      # keep 30 neighbours per movie

At runtime the app loads the table with ``load_neighbor_index()`` and calls
``index.similar(tmdb_id)``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
CATALOGUE_PATH = BASE_DIR / "data" / "movies.csv.gz"
NEIGHBORS_PATH = BASE_DIR / "data" / "cf_neighbors.csv.gz"

# Movies with very few ratings produce noisy, unreliable neighbours, so drop them.
MIN_RATINGS_PER_MOVIE = 10
# Neighbours weaker than this are co-rated by too few people to trust.
MIN_SIMILARITY = 0.10
DEFAULT_TOP_K = 20


# --------------------------------------------------------------------------- runtime
@dataclass
class NeighborIndex:
    """In-memory lookup of precomputed item-based CF neighbours. Tiny + fast."""

    neighbors: dict[int, list[tuple[int, float]]]

    def __contains__(self, tmdb_id: object) -> bool:
        return int(tmdb_id) in self.neighbors

    @property
    def covered_count(self) -> int:
        return len(self.neighbors)

    def similar(self, tmdb_id: int, limit: int = 12) -> list[tuple[int, float]]:
        """Return up to ``limit`` (tmdb_id, similarity) pairs most similar to ``tmdb_id``."""
        return self.neighbors.get(int(tmdb_id), [])[:limit]


def load_neighbor_index(path: str | Path = NEIGHBORS_PATH) -> NeighborIndex:
    """Load the shipped neighbour table. Returns an empty index if it's missing."""
    path = Path(path)
    if not path.exists():
        return NeighborIndex(neighbors={})
    table = pd.read_csv(path)
    neighbors: dict[int, list[tuple[int, float]]] = {}
    for movie_id, group in table.groupby("movie_id", sort=False):
        ordered = group.sort_values("score", ascending=False)
        neighbors[int(movie_id)] = list(
            zip(ordered["neighbor_id"].astype(int), ordered["score"].astype(float))
        )
    return NeighborIndex(neighbors=neighbors)


# ----------------------------------------------------------------------------- build
def _load_ratings(archive: Path, small: bool) -> pd.DataFrame:
    ratings_name = "ratings_small.csv" if small else "ratings.csv"
    links_name = "links_small.csv" if small else "links.csv"
    ratings_path = archive / ratings_name
    links_path = archive / links_name
    if not ratings_path.exists() or not links_path.exists():
        raise FileNotFoundError(
            f"Need {ratings_path} and {links_path}. The raw ratings CSVs are not committed "
            "to git; download the Kaggle 'The Movies Dataset' into archive/ first."
        )

    links = pd.read_csv(links_path, usecols=["movieId", "tmdbId"]).dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    ratings = pd.read_csv(ratings_path, usecols=["userId", "movieId", "rating"])
    ratings = ratings.merge(links, on="movieId", how="inner")
    ratings = ratings.rename(columns={"tmdbId": "id"})
    return ratings[["userId", "id", "rating"]]


def build_neighbor_table(
    archive: Path = ARCHIVE_DIR,
    catalogue_path: Path = CATALOGUE_PATH,
    *,
    small: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """Train item-based CF and return a tidy (movie_id, neighbor_id, score) table."""
    ratings = _load_ratings(archive, small)

    # Restrict to movies that actually exist in the shipped catalogue.
    catalogue_ids = set(pd.read_csv(catalogue_path, usecols=["id"])["id"].astype(int))
    ratings = ratings[ratings["id"].isin(catalogue_ids)]

    # Drop thinly-rated movies that would only add noise.
    counts = ratings.groupby("id")["id"].transform("size")
    ratings = ratings[counts >= MIN_RATINGS_PER_MOVIE]
    if ratings.empty:
        raise ValueError("No movies left after filtering — check the rating data.")

    # Adjusted cosine: subtract each user's mean rating to remove per-user bias.
    user_means = ratings.groupby("userId")["rating"].transform("mean")
    ratings = ratings.assign(centered=(ratings["rating"] - user_means).astype(np.float32))

    item_codes = ratings["id"].astype("category")
    user_codes = ratings["userId"].astype("category")
    item_ids = item_codes.cat.categories.to_numpy(dtype=np.int64)
    n_items = item_ids.size
    n_users = user_codes.cat.categories.size

    # Sparse item x user matrix, then L2-normalise rows so dot product == cosine.
    item_user = csr_matrix(
        (
            ratings["centered"].to_numpy(),
            (item_codes.cat.codes.to_numpy(), user_codes.cat.codes.to_numpy()),
        ),
        shape=(n_items, n_users),
        dtype=np.float32,
    )
    item_user = normalize(item_user, norm="l2", axis=1, copy=False)

    print(f"Computing item-item similarity for {n_items:,} movies x {n_users:,} users…")

    rows: list[int] = []
    neighbors: list[int] = []
    scores: list[float] = []
    keep = top_k + 1  # +1 because a movie's nearest neighbour is itself.
    batch = 512
    for start in range(0, n_items, batch):
        stop = min(start + batch, n_items)
        # (batch x users) @ (users x items) -> dense (batch x items) cosine block.
        block = (item_user[start:stop] @ item_user.T).toarray()
        for offset in range(stop - start):
            row = start + offset
            sims = block[offset]
            sims[row] = -1.0  # never recommend the movie itself
            top = np.argpartition(sims, -keep)[-keep:]
            top = top[np.argsort(sims[top])[::-1]]
            taken = 0
            for col in top:
                value = float(sims[col])
                if value < MIN_SIMILARITY or taken >= top_k:
                    continue
                rows.append(int(item_ids[row]))
                neighbors.append(int(item_ids[col]))
                scores.append(round(value, 4))
                taken += 1
        print(f"  {stop:,}/{n_items:,} movies processed", end="\r")
    print()

    return pd.DataFrame({"movie_id": rows, "neighbor_id": neighbors, "score": scores})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--out", type=Path, default=NEIGHBORS_PATH)
    parser.add_argument("--small", action="store_true",
                        help="Use ratings_small.csv / links_small.csv (faster, less coverage).")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    table = build_neighbor_table(args.archive, small=args.small, top_k=args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False, compression="gzip")
    print(
        f"Wrote {len(table):,} neighbour pairs for {table['movie_id'].nunique():,} movies "
        f"-> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
