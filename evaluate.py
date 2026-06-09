"""Held-out ranking evaluation for the collaborative-filtering recommender.

Unlike the lightweight proxy checks documented in ``docs/evaluation.md``, this is a real
train/test split on the MovieLens ratings: we hide a portion of every user's ratings,
build item-based CF on the *training* portion only, then measure whether the recommender
ranks each user's held-out, highly-rated movies near the top.

Protocol
--------
1. Load MovieLens ratings, map ``movieId`` -> catalogue TMDB ``id`` via ``links``.
2. Per-user random split: ~20% of each user's ratings become the **test** set, the rest
   are **train** (seed-controlled, so runs are reproducible).
3. Build item-item adjusted-cosine CF from the **train** ratings only — no leakage.
4. For each user, score candidate movies by summing neighbour similarities from the
   movies they liked in train, excluding anything they already rated in train.
5. Relevant items = the movies a user rated ``>= 4.0`` in the held-out **test** set.
6. Report Precision@K, Recall@K, NDCG@K and MAP@K, averaged over evaluable users, for
   the CF model and a most-popular baseline.

Usage
-----
    python evaluate.py                 # ratings_small.csv (fast: ~671 users)
    python evaluate.py --full          # ratings.csv (slow, full coverage)
    python evaluate.py --k 10 --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from collaborative import MIN_RATINGS_PER_MOVIE, MIN_SIMILARITY

BASE_DIR = Path(__file__).resolve().parent
ARCHIVE_DIR = BASE_DIR / "archive"
CATALOGUE_PATH = BASE_DIR / "data" / "movies.csv.gz"

# A rating at or above this counts as a positive ("the user liked it").
LIKE_THRESHOLD = 4.0
TEST_FRACTION = 0.2
DEFAULT_K = 10
DEFAULT_TOP_K = 20  # neighbours kept per movie when building CF


# --------------------------------------------------------------------------- data
def load_ratings(archive: Path, full: bool) -> pd.DataFrame:
    ratings_name = "ratings.csv" if full else "ratings_small.csv"
    links_name = "links.csv" if full else "links_small.csv"
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
    ratings = ratings.merge(links, on="movieId", how="inner").rename(columns={"tmdbId": "id"})

    catalogue_ids = set(pd.read_csv(CATALOGUE_PATH, usecols=["id"])["id"].astype(int))
    ratings = ratings[ratings["id"].isin(catalogue_ids)]
    return ratings[["userId", "id", "rating"]].reset_index(drop=True)


def train_test_split_per_user(ratings: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly hold out TEST_FRACTION of each user's ratings for the test set."""
    rng = np.random.default_rng(seed)
    is_test = np.zeros(len(ratings), dtype=bool)
    for _, idx in ratings.groupby("userId").groups.items():
        idx = np.asarray(idx)
        n_test = int(round(len(idx) * TEST_FRACTION))
        if n_test == 0:
            continue
        is_test[rng.choice(idx, size=n_test, replace=False)] = True
    return ratings[~is_test].copy(), ratings[is_test].copy()


# ----------------------------------------------------------------------- CF model
def build_neighbors(train: pd.DataFrame, top_k: int) -> dict[int, list[tuple[int, float]]]:
    """Item-based adjusted-cosine CF from training ratings only. Mirrors collaborative.py."""
    counts = train.groupby("id")["id"].transform("size")
    train = train[counts >= MIN_RATINGS_PER_MOVIE]
    if train.empty:
        raise ValueError("No movies left after the min-ratings filter — check the split.")

    user_means = train.groupby("userId")["rating"].transform("mean")
    centered = (train["rating"] - user_means).astype(np.float32)

    item_codes = train["id"].astype("category")
    user_codes = train["userId"].astype("category")
    item_ids = item_codes.cat.categories.to_numpy(dtype=np.int64)
    n_items = item_ids.size
    n_users = user_codes.cat.categories.size

    item_user = csr_matrix(
        (centered.to_numpy(), (item_codes.cat.codes.to_numpy(), user_codes.cat.codes.to_numpy())),
        shape=(n_items, n_users),
        dtype=np.float32,
    )
    item_user = normalize(item_user, norm="l2", axis=1, copy=False)

    neighbors: dict[int, list[tuple[int, float]]] = {}
    keep = top_k + 1  # +1 because a movie's nearest neighbour is itself
    batch = 512
    for start in range(0, n_items, batch):
        stop = min(start + batch, n_items)
        block = (item_user[start:stop] @ item_user.T).toarray()
        for offset in range(stop - start):
            row = start + offset
            sims = block[offset]
            sims[row] = -1.0
            top = np.argpartition(sims, -keep)[-keep:]
            top = top[np.argsort(sims[top])[::-1]]
            picks = [
                (int(item_ids[col]), float(sims[col]))
                for col in top
                if sims[col] >= MIN_SIMILARITY
            ][:top_k]
            if picks:
                neighbors[int(item_ids[row])] = picks
    return neighbors


# -------------------------------------------------------------------- recommend
def recommend_cf(profile: list[int], neighbors: dict, seen: set[int], k: int) -> list[int]:
    """Score candidates by summing neighbour similarities from the user's liked movies."""
    scores: dict[int, float] = {}
    for movie_id in profile:
        for neighbor_id, sim in neighbors.get(movie_id, []):
            if neighbor_id in seen:
                continue
            scores[neighbor_id] = scores.get(neighbor_id, 0.0) + sim
    ranked = sorted(scores, key=scores.get, reverse=True)
    return ranked[:k]


def recommend_popular(popularity: list[int], seen: set[int], k: int) -> list[int]:
    out = []
    for movie_id in popularity:
        if movie_id in seen:
            continue
        out.append(movie_id)
        if len(out) >= k:
            break
    return out


# ----------------------------------------------------------------------- metrics
def ranking_metrics(recommended: list[int], relevant: set[int], k: int) -> dict[str, float]:
    hits = [1 if movie_id in relevant else 0 for movie_id in recommended[:k]]
    n_hits = sum(hits)

    precision = n_hits / k
    recall = n_hits / len(relevant)

    dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
    idcg = sum(1 / np.log2(rank + 2) for rank in range(min(len(relevant), k)))
    ndcg = dcg / idcg if idcg else 0.0

    ap, running = 0.0, 0
    for rank, hit in enumerate(hits, start=1):
        if hit:
            running += 1
            ap += running / rank
    map_score = ap / min(len(relevant), k)

    return {"precision": precision, "recall": recall, "ndcg": ndcg, "map": map_score}


# -------------------------------------------------------------------------- main
def evaluate(ratings: pd.DataFrame, k: int, seed: int) -> dict:
    train, test = train_test_split_per_user(ratings, seed)
    neighbors = build_neighbors(train, DEFAULT_TOP_K)

    # Most-popular baseline = movies with the most training ratings.
    popularity = train["id"].value_counts().index.astype(int).tolist()

    train_liked = train[train["rating"] >= LIKE_THRESHOLD].groupby("userId")["id"].apply(list)
    train_seen = train.groupby("userId")["id"].apply(set)
    test_liked = test[test["rating"] >= LIKE_THRESHOLD].groupby("userId")["id"].apply(set)

    cf_totals = {m: 0.0 for m in ("precision", "recall", "ndcg", "map")}
    pop_totals = {m: 0.0 for m in ("precision", "recall", "ndcg", "map")}
    evaluated = 0

    for user_id, relevant in test_liked.items():
        profile = train_liked.get(user_id, [])
        if not profile or not relevant:
            continue
        seen = train_seen.get(user_id, set())

        cf_recs = recommend_cf(profile, neighbors, seen, k)
        if not cf_recs:  # user's liked movies have no CF neighbours — not scorable
            continue

        pop_recs = recommend_popular(popularity, seen, k)
        for name, value in ranking_metrics(cf_recs, relevant, k).items():
            cf_totals[name] += value
        for name, value in ranking_metrics(pop_recs, relevant, k).items():
            pop_totals[name] += value
        evaluated += 1

    cf = {m: v / evaluated for m, v in cf_totals.items()}
    pop = {m: v / evaluated for m, v in pop_totals.items()}
    return {
        "k": k,
        "users_total": ratings["userId"].nunique(),
        "users_evaluated": evaluated,
        "train_ratings": len(train),
        "test_ratings": len(test),
        "cf_movies": len(neighbors),
        "cf": cf,
        "popularity": pop,
    }


def report(result: dict) -> None:
    k = result["k"]
    print("Held-out CF ranking evaluation")
    print("==============================")
    print(f"Users (total / evaluable):     {result['users_total']:,} / {result['users_evaluated']:,}")
    print(f"Ratings (train / test):        {result['train_ratings']:,} / {result['test_ratings']:,}")
    print(f"Movies with CF neighbours:     {result['cf_movies']:,}")
    print()
    header = f"{'Model':<18}{'Precision@'+str(k):>14}{'Recall@'+str(k):>13}{'NDCG@'+str(k):>11}{'MAP@'+str(k):>10}"
    print(header)
    print("-" * len(header))
    for label, key in (("Item-based CF", "cf"), ("Most-popular", "popularity")):
        m = result[key]
        print(f"{label:<18}{m['precision']:>14.3f}{m['recall']:>13.3f}{m['ndcg']:>11.3f}{m['map']:>10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_DIR)
    parser.add_argument("--full", action="store_true", help="Use ratings.csv (slow, full coverage).")
    parser.add_argument("--k", type=int, default=DEFAULT_K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ratings = load_ratings(args.archive, args.full)
    result = evaluate(ratings, args.k, args.seed)
    report(result)


if __name__ == "__main__":
    main()
