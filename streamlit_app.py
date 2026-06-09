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

import streamlit as st

from collaborative import load_neighbor_index
from engine import build_engine

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
