"""Movie Finder — a Streamlit app that recommends movies from a free-text description.

Describe the kind of movie you want in plain English and the classical-ML engine in
``engine.py`` ranks ~45k movies (TMDB / MovieLens dataset) to match your intent.

Two complementary recommenders work together:
- **Content-based** (``engine.py``): TF-IDF matches your text to movie metadata.
- **Collaborative filtering** (``collaborative.py``): "More like this" surfaces movies
  that MovieLens users co-rated with a title you picked.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub and point Streamlit Community Cloud at streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from collaborative import load_neighbor_index
from engine import DEFAULT_DATA_PATH, build_engine

SAMPLE_QUERIES = [
    "A hero sacrifices himself to save the planet, but his own team betrays him.",
    "A scary haunted-house ghost story that keeps me on edge.",
    "A funny romantic comedy about falling in love in Paris.",
    "An animated family movie with talking animals for kids.",
    "A gritty crime thriller about a detective hunting a serial killer.",
]

st.set_page_config(
    page_title="Movie Finder",
    page_icon="🎬",
    layout="wide",
)


@st.cache_resource(show_spinner="Building the recommendation model (first load only)…")
def get_engine():
    """Build the TF-IDF engine once and reuse it across reruns and sessions."""
    # Bump this token whenever engine output shape changes, to invalidate the
    # cached resource on deploy (e.g. when the "id" field was added for CF).
    _cache_version = "cf-v1"
    return build_engine()


@st.cache_resource(show_spinner=False)
def get_neighbor_index():
    """Load the precomputed collaborative-filtering neighbour table once."""
    return load_neighbor_index()


@st.cache_data(show_spinner=False)
def load_catalogue(data_path: str) -> pd.DataFrame:
    """Load the processed movie catalogue for EDA and reporting views."""
    return pd.read_csv(data_path)


@st.cache_data(show_spinner=False)
def load_cf_table(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["movie_id", "neighbor_id", "score"])
    return pd.read_csv(path)


def split_genres(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .str.split(", ")
        .explode()
        .str.strip()
        .replace("", np.nan)
        .dropna()
    )


def render_movie(movie: dict, neighbor_index) -> None:
    with st.container(border=True):
        if movie["poster"]:
            st.image(movie["poster"], use_container_width=True)
        else:
            st.markdown(
                "<div style='height:330px;display:flex;align-items:center;"
                "justify-content:center;background:#1e2128;border-radius:8px;"
                "color:#888;'>No poster</div>",
                unsafe_allow_html=True,
            )
        st.markdown(f"**{movie['title']}**  ·  {movie['year']}")
        st.progress(min(movie["match"], 100.0) / 100.0, text=f"{movie['match']:.0f}% match")
        st.caption(f"🎭 {movie['genre']}")
        meta = []
        if movie["rating"]:
            meta.append(f"⭐ {movie['rating']} ({movie['votes']:,} votes)")
        if movie["language"]:
            meta.append(f"🗣️ {str(movie['language']).upper()}")
        if meta:
            st.caption("  ·  ".join(meta))
        if movie["director"]:
            st.caption(f"🎬 Dir: {movie['director']}")
        if movie["cast"]:
            st.caption(f"👥 {movie['cast']}")
        if movie["overview"]:
            with st.expander("Overview"):
                st.write(movie["overview"])
                if movie["keywords"]:
                    st.caption(f"Keywords: {movie['keywords']}")
        # Collaborative-filtering entry point: only offer it where we have neighbours.
        movie_id = movie.get("id")
        if movie_id is not None and movie_id in neighbor_index:
            if st.button("🎯 More like this", key=f"similar-{movie_id}", use_container_width=True):
                st.session_state.similar_to = {"id": movie_id, "title": movie["title"]}
                st.rerun()


def render_grid(results: list[dict]) -> None:
    columns_per_row = 4
    for start in range(0, len(results), columns_per_row):
        row = results[start : start + columns_per_row]
        cols = st.columns(columns_per_row)
        for col, movie in zip(cols, row):
            with col:
                render_movie(movie, neighbor_index)


def render_eda(catalogue: pd.DataFrame, cf_table: pd.DataFrame) -> None:
    st.subheader("Exploratory Data Analysis")

    total_movies = len(catalogue)
    year_min = int(catalogue["Release_Year"].replace(0, np.nan).min())
    year_max = int(catalogue["Release_Year"].max())
    poster_rate = catalogue["Poster_Url"].fillna("").ne("").mean() * 100
    cf_coverage = cf_table["movie_id"].nunique() if not cf_table.empty else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Movies", f"{total_movies:,}")
    metric_cols[1].metric("Release years", f"{year_min}-{year_max}")
    metric_cols[2].metric("Poster coverage", f"{poster_rate:.1f}%")
    metric_cols[3].metric("CF coverage", f"{cf_coverage:,}")

    st.divider()
    left, right = st.columns(2)

    with left:
        genre_counts = split_genres(catalogue["Genre"]).value_counts().head(15)
        st.caption("Top genres")
        st.bar_chart(genre_counts)

        language_counts = catalogue["original_language"].fillna("unknown").str.upper().value_counts().head(12)
        st.caption("Top original languages")
        st.bar_chart(language_counts)

    with right:
        by_year = (
            catalogue[catalogue["Release_Year"] > 0]
            .groupby("Release_Year")
            .size()
            .rename("movies")
        )
        st.caption("Movies by release year")
        st.line_chart(by_year)

        rating_by_year = (
            catalogue[(catalogue["Release_Year"] > 0) & (catalogue["vote_count"] >= 50)]
            .groupby("Release_Year")["vote_average"]
            .mean()
            .rename("average rating")
        )
        st.caption("Average TMDB rating by year, minimum 50 votes")
        st.line_chart(rating_by_year)

    st.divider()
    st.caption("Highest-rated movies with at least 1,000 TMDB votes")
    top_rated = (
        catalogue[catalogue["vote_count"] >= 1000]
        .sort_values(["vote_average", "vote_count"], ascending=False)
        [["title", "Release_Year", "Genre", "vote_average", "vote_count", "user_rating_mean"]]
        .head(15)
    )
    st.dataframe(top_rated, use_container_width=True, hide_index=True)


def evaluate_content_engine(engine) -> pd.DataFrame:
    benchmark = [
        {
            "query": "scary haunted house ghost story",
            "expected_genres": {"Horror"},
        },
        {
            "query": "funny romantic comedy about falling in love",
            "expected_genres": {"Comedy", "Romance"},
        },
        {
            "query": "space adventure alien invasion save earth",
            "expected_genres": {"Science Fiction", "Adventure", "Action"},
        },
        {
            "query": "detective hunting a serial killer crime thriller",
            "expected_genres": {"Crime", "Thriller", "Mystery"},
        },
        {
            "query": "animated family movie with talking animals",
            "expected_genres": {"Animation", "Family"},
        },
    ]

    rows = []
    for case in benchmark:
        results = engine.recommend(case["query"], limit=10)
        hits = 0
        top_titles = []
        for movie in results:
            genres = set(str(movie["genre"]).split(", "))
            if genres.intersection(case["expected_genres"]):
                hits += 1
            top_titles.append(movie["title"])
        rows.append(
            {
                "query": case["query"],
                "expected_genres": ", ".join(sorted(case["expected_genres"])),
                "precision@10": hits / 10,
                "top_result": top_titles[0] if top_titles else "",
                "top_3_results": " | ".join(top_titles[:3]),
            }
        )
    return pd.DataFrame(rows)


def render_evaluation(engine, catalogue: pd.DataFrame, cf_table: pd.DataFrame) -> None:
    st.subheader("Evaluation Results")
    st.caption(
        "These are lightweight app-level checks, not a full offline recommender evaluation. "
        "The shipped app does not include a labelled relevance test set."
    )

    content_eval = evaluate_content_engine(engine)
    average_precision = content_eval["precision@10"].mean()
    cf_covered = cf_table["movie_id"].nunique() if not cf_table.empty else 0
    cf_coverage_rate = cf_covered / len(catalogue) if len(catalogue) else 0
    avg_neighbors = cf_table.groupby("movie_id").size().mean() if not cf_table.empty else 0
    avg_similarity = cf_table["score"].mean() if not cf_table.empty else 0

    cols = st.columns(4)
    cols[0].metric("Content precision@10", f"{average_precision:.2f}")
    cols[1].metric("CF covered movies", f"{cf_covered:,}")
    cols[2].metric("CF coverage rate", f"{cf_coverage_rate:.1%}")
    cols[3].metric("Avg CF neighbors", f"{avg_neighbors:.1f}")

    st.caption(f"Average stored collaborative-filtering similarity: {avg_similarity:.3f}")
    st.dataframe(content_eval, use_container_width=True, hide_index=True)

    if not cf_table.empty:
        neighbor_distribution = cf_table.groupby("movie_id").size().value_counts().sort_index()
        st.caption("Collaborative-filtering neighbour count distribution")
        st.bar_chart(neighbor_distribution)


def render_deployment_architecture() -> None:
    st.subheader("Deployment Architecture")
    st.markdown(
        """
        **Runtime flow**

        1. User opens the Streamlit app in a browser.
        2. Streamlit serves `streamlit_app.py`.
        3. `get_engine()` builds and caches the TF-IDF recommender from `data/movies.csv.gz`.
        4. `get_neighbor_index()` loads the precomputed collaborative-filtering table from `data/cf_neighbors.csv.gz`.
        5. Text search calls `engine.recommend(...)`; "More like this" calls `engine.similar(...)`.
        6. Results render as movie cards with metadata and poster URLs from the TMDB image CDN.

        **Offline build flow**

        1. Raw Kaggle/TMDB files are kept locally in `archive/`.
        2. `build_dataset.py` creates the deployable catalogue `data/movies.csv.gz`.
        3. `collaborative.py` computes item-item similarities and writes `data/cf_neighbors.csv.gz`.
        4. Only the small compressed artifacts are deployed with the app.
        """
    )
    st.code(
        """
Browser
  -> Streamlit app: streamlit_app.py
      -> Content model: engine.py
          -> data/movies.csv.gz
          -> TF-IDF word + char matrices cached in memory
      -> Collaborative lookup: collaborative.py
          -> data/cf_neighbors.csv.gz
      -> Movie posters
          -> TMDB image CDN

Offline data preparation
  archive/*.csv
      -> build_dataset.py
          -> data/movies.csv.gz
      -> collaborative.py
          -> data/cf_neighbors.csv.gz
        """.strip(),
        language="text",
    )


# ----------------------------------------------------------------------------- UI
st.title("🎬 Movie Finder")
st.write(
    "Describe the movie you're in the mood for — plot, vibe, genre, anything — "
    "and get the closest matches."
)

if "query" not in st.session_state:
    st.session_state.query = SAMPLE_QUERIES[0]

with st.sidebar:
    st.header("Try an example")
    for example in SAMPLE_QUERIES:
        if st.button(example, use_container_width=True):
            st.session_state.query = example
    st.divider()
    num_results = st.slider("Number of results", min_value=4, max_value=24, value=12, step=2)
    st.caption(
        "Dataset: TMDB metadata + MovieLens ratings. "
        "Posters via the TMDB image CDN."
    )

# Load the models up front so the first search is instant.
engine = get_engine()
neighbor_index = get_neighbor_index()
catalogue = load_catalogue(str(DEFAULT_DATA_PATH))
cf_table = load_cf_table(str(Path(__file__).resolve().parent / "data" / "cf_neighbors.csv.gz"))

search_tab, eda_tab, evaluation_tab, architecture_tab = st.tabs(
    ["Search", "EDA", "Evaluation", "Deployment Architecture"]
)

with search_tab:
    query = st.text_area(
        "What kind of movie do you want to watch?",
        key="query",
        height=120,
        placeholder="e.g. A space adventure where a small crew has to save Earth from an alien invasion.",
    )

    search = st.button("🔍 Find movies", type="primary")

    # Collaborative filtering: "More like this" takes priority over a text search.
    if st.session_state.get("similar_to"):
        seed = st.session_state.similar_to
        header, clear = st.columns([4, 1])
        with header:
            st.subheader(f"🎯 Because you liked *{seed['title']}*")
            st.caption("Movies MovieLens users tended to rate similarly (collaborative filtering).")
        with clear:
            if st.button("← Back to search", use_container_width=True):
                del st.session_state.similar_to
                st.rerun()

        similar = engine.similar(seed["id"], neighbor_index, limit=num_results)
        if similar:
            render_grid(similar)
        else:
            st.info("Not enough rating overlap to recommend similar titles for this one.")

    elif search or query:
        cleaned = query.strip()
        if len(cleaned) < 8:
            st.warning("Please describe the movie in at least 8 characters.")
        else:
            results = engine.recommend(cleaned, limit=num_results)
            st.subheader(f"Top {len(results)} matches")
            render_grid(results)

with eda_tab:
    render_eda(catalogue, cf_table)

with evaluation_tab:
    render_evaluation(engine, catalogue, cf_table)

with architecture_tab:
    render_deployment_architecture()
