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

## Limitations

- This is not a train/test split evaluation.
- The content metric checks genre alignment, not human judgement of relevance.
- Collaborative-filtering quality is summarized by coverage and stored similarity because
  raw full ratings data is not shipped with the deployed app.
