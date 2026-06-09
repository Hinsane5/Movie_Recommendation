# Deployment Architecture

## Runtime Flow

1. User opens the Streamlit app in a browser.
2. Streamlit serves `streamlit_app.py`.
3. `get_engine()` builds and caches the TF-IDF recommender from `data/movies.csv.gz`.
4. `get_neighbor_index()` loads the precomputed collaborative-filtering table from
   `data/cf_neighbors.csv.gz`.
5. Text search calls `engine.recommend(...)`.
6. "More like this" calls `engine.similar(...)`.
7. Movie cards render metadata from the local catalogue and poster images from the TMDB CDN.

```text
Browser
  -> Streamlit app: streamlit_app.py
      -> Content model: engine.py
          -> data/movies.csv.gz
          -> TF-IDF word + char matrices cached in memory
      -> Collaborative lookup: collaborative.py
          -> data/cf_neighbors.csv.gz
      -> Movie posters
          -> TMDB image CDN
```

## Offline Build Flow

1. Raw Kaggle/TMDB files are kept locally in `archive/`.
2. `build_dataset.py` creates the deployable catalogue `data/movies.csv.gz`.
3. `collaborative.py` computes item-item similarities and writes
   `data/cf_neighbors.csv.gz`.
4. Only the compressed artifacts in `data/` are deployed with the app.

```text
archive/*.csv
  -> build_dataset.py
      -> data/movies.csv.gz
  -> collaborative.py
      -> data/cf_neighbors.csv.gz
```

## Deployment Target

The Streamlit Cloud app should point to:

- Repository: `Hinsane5/Movie_Recommendation`
- Branch: `main`
- Main file: `streamlit_app.py`

Pushing to `main` triggers a Streamlit Cloud redeploy when the app is connected to this
repository and branch.
