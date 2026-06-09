# Evaluation Results

These are lightweight app-level checks for the shipped recommender. The repository does
not include a labelled relevance test set, so the content-based result below uses
genre-intent proxy queries.

## Content-Based Recommender

Method:

1. Run five representative free-text queries through `engine.recommend(..., limit=10)`.
2. Define expected genres for each query.
3. Count a hit when a returned movie has at least one expected genre.
4. Report `precision@10` per query and the average.

| Query | Expected Genres | Precision@10 | Top Result |
|---|---|---:|---|
| scary haunted house ghost story | Horror | 1.00 | The Innocents |
| funny romantic comedy about falling in love | Comedy, Romance | 1.00 | Pyaar Ka Punchnama 2 |
| space adventure alien invasion save earth | Action, Adventure, Science Fiction | 1.00 | The War in Space |
| detective hunting a serial killer crime thriller | Crime, Mystery, Thriller | 1.00 | Night Game |
| animated family movie with talking animals | Animation, Family | 1.00 | Wow! A Talking Fish! |

Average content `precision@10`: **1.00**

## Collaborative Filtering

The collaborative-filtering table is precomputed in `data/cf_neighbors.csv.gz`.

| Metric | Result |
|---|---:|
| Covered movies | 19,779 |
| Coverage rate over catalogue | 43.8% |
| Average neighbours per covered movie | 16.61 |
| Average stored similarity score | 0.217 |

## Held-Out Ranking Evaluation

A real train/test split on the MovieLens ratings (`evaluate.py`). For each user, ~20% of
their ratings are hidden as a test set; item-based CF is rebuilt on the **training**
portion only, then we measure whether it ranks each user's held-out, highly-rated
(`>= 4.0`) movies near the top. A most-popular ranking is included as a baseline.

Run on `ratings_small.csv` (671 users, 99,809 catalogue-matched ratings), `K = 10`,
`seed = 42`:

| Model | Precision@10 | Recall@10 | NDCG@10 | MAP@10 |
|---|---:|---:|---:|---:|
| Item-based CF | 0.065 | 0.076 | 0.096 | 0.047 |
| Most-popular | 0.111 | 0.097 | 0.150 | 0.084 |

Evaluable users: 653 / 671. Train/test ratings: 79,848 / 19,961. Movies with CF
neighbours after the min-ratings filter: 1,919.

**Reading the result:** on this small split, the popularity baseline beats memory-based
CF on every metric. This is a well-known effect — most-popular is a notoriously strong
baseline, and on MovieLens-small the held-out positives are dominated by widely-rated
titles, while CF can only rank the 1,919 movies that survive the `MIN_RATINGS_PER_MOVIE`
filter. CF's value is personalisation and discovery ("users who liked X also liked Y"),
which raw precision against popular holdouts under-credits. The fuller `ratings.csv`
(`python evaluate.py --full`) gives CF far more co-rating signal and coverage.

Reproduce with:

```bash
python3 evaluate.py            # ratings_small.csv (fast)
python3 evaluate.py --full     # ratings.csv (slow, full coverage)
```

## Limitations

- The held-out split uses explicit MovieLens ratings as ground truth — a user not
  re-rating a recommended movie is treated as "not relevant," which understates real
  precision (missing-not-at-random).
- The content metric checks genre alignment, not human judgement of relevance, and is not
  part of the held-out split (content search takes a free-text query, not a user profile).
- Collaborative-filtering coverage is bounded by `MIN_RATINGS_PER_MOVIE`; thinly-rated
  movies are intentionally excluded as noisy.
