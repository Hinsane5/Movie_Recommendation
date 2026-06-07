# 🎬 Movie Finder

Describe the kind of movie you're in the mood for — plot, mood, genre, anything — and get
the closest matches from **~45,000 films**. Built with **Streamlit** and classical machine
learning only. No login, no deep learning.

**Live demo:** deploy in one click on [Streamlit Community Cloud](https://share.streamlit.io)
(see below).

## How it works

The recommender (`engine.py`) ranks every movie against your description using:

- **Word TF-IDF (1–2 grams)** over each movie's title, genre, keywords, tagline, overview,
  cast, and director — weighted so titles/genres/keywords matter most.
- **Character TF-IDF (3–5 grams)** so the search tolerates typos and spelling variants.
- **Cosine similarity** between your description and every movie.
- **Query expansion** with movie-domain synonyms (hero → superhero, betray → traitor, …).
- **Genre-intent & keyword-overlap** boosts toward what you actually described.
- A **concept-intent scorer** that recognises stronger ideas — superhero teams, betrayal,
  sacrifice, save-the-world, ghosts, comedy, family animation — precomputed over the corpus
  for fast queries.
- **Rating, vote count, and popularity** (TMDB) plus a **MovieLens user-rating** signal as
  gentle tie-breakers.

No neural network is involved — it's interpretable, fast (~0.1 s/query), and fits within a
free Streamlit container.

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open the URL Streamlit prints (default http://localhost:8501).

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pick this repo, branch `main`, main file `streamlit_app.py`.
4. Deploy. The first load builds and caches the TF-IDF model (~1 minute); every load after
   is instant.

Everything the app needs is committed — the slim catalogue lives in
`data/movies.csv.gz` (~14 MB), so there's nothing else to configure.

## Dataset

Source: [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)
(TMDB metadata + MovieLens ratings). The full raw CSVs are **gigabytes** and are *not*
committed (see `.gitignore`). The committed `data/movies.csv.gz` is a slim, preprocessed
catalogue.

To regenerate it from the raw `archive/` CSVs:

```bash
python build_dataset.py
```

## Project layout

```
streamlit_app.py      Streamlit UI
engine.py             Recommendation engine (TF-IDF + cosine + intent scoring)
build_dataset.py      Rebuilds data/movies.csv.gz from the raw Kaggle CSVs
data/movies.csv.gz    Slim preprocessed catalogue (committed, ~14 MB)
requirements.txt      Python dependencies
.streamlit/config.toml  Theme + server config
```
