"""Classical-ML movie recommendation engine.

Given a free-text paragraph describing the kind of movie a user wants, this engine
ranks the catalogue using:

- Word TF-IDF (1-2 grams) for semantic overlap between the query and each movie's
  title / genre / keywords / tagline / overview / cast / director.
- Character TF-IDF (3-5 grams) for typo tolerance and spelling variation.
- Query expansion with movie-domain synonyms (hero -> superhero, betray -> traitor, ...).
- Genre-intent and keyword-overlap boosts.
- A concept-intent scorer (superhero teams, betrayal, sacrifice, save-the-world, ...)
  precomputed once over the corpus for fast queries.
- Rating / vote-count / popularity quality used as gentle tie-breakers.

No deep learning is used. The model is built once from ``data/movies.csv.gz`` and is
designed to fit comfortably within Streamlit Community Cloud's memory limits
(capped vocabularies + float32 sparse matrices).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "movies.csv.gz"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

TOP_RESULTS = 12
RERANK_CANDIDATES = 1200

# Keep the runtime footprint friendly for a 1 GB Streamlit container.
WORD_MAX_FEATURES = 80_000
CHAR_MAX_FEATURES = 40_000

IMPORTANT_STOPWORDS = {
    "about", "after", "because", "before", "being", "could", "their", "there",
    "these", "those", "through", "movie", "movies", "watch", "where", "would",
    "want", "waant",
}

QUERY_EXPANSIONS = {
    "hero": "superhero savior brave champion protagonist rescue protect",
    "heroes": "superheroes saviors team rescue protect",
    "sacrifice": "sacrifices sacrificial gives life save protect",
    "sacrifices": "sacrifice sacrificial gives life save protect",
    "betray": "betrayed betrayal traitor deception double cross",
    "betrays": "betrayed betrayal traitor deception double cross",
    "team": "group squad allies friends crew",
    "planet": "world earth galaxy universe space",
    "space": "planet galaxy universe alien science fiction",
    "scary": "horror terrifying fear haunted monster",
    "funny": "comedy humorous hilarious joke",
    "romantic": "romance love relationship couple",
}

GENRE_ALIASES = {
    "Action": {"action", "fight", "battle", "hero", "superhero", "war", "revenge", "combat"},
    "Adventure": {"adventure", "journey", "quest", "explore", "rescue", "mission", "survive"},
    "Animation": {"animation", "animated", "anime", "cartoon"},
    "Comedy": {"comedy", "funny", "humor", "hilarious", "joke"},
    "Crime": {"crime", "criminal", "detective", "mafia", "police", "murder"},
    "Drama": {"drama", "emotional", "family", "life", "relationship"},
    "Fantasy": {"fantasy", "magic", "wizard", "myth", "dragon", "supernatural"},
    "Horror": {"horror", "scary", "haunted", "ghost", "monster", "terrifying"},
    "Mystery": {"mystery", "secret", "detective", "clue", "investigate"},
    "Romance": {"romance", "romantic", "love", "couple", "relationship"},
    "Science Fiction": {"science", "sci", "sci-fi", "space", "alien", "planet", "future", "robot"},
    "Thriller": {"thriller", "suspense", "tense", "danger", "kidnap", "escape"},
}

CONCEPT_RULES = {
    "superhero": {
        "query_terms": {"hero", "heroes", "heros", "superhero", "superheroes", "superman", "avenger"},
        "movie_terms": {
            "anti hero", "based on comic", "child hero", "crime fighter", "dc comics",
            "marvel comic", "super powers", "superhero", "superhero team", "superhuman",
        },
        "weight": 0.055,
    },
    "team": {
        "query_terms": {"team", "squad", "group", "crew", "allies", "ally", "friends"},
        "movie_terms": {
            "avengers", "crew", "justice league", "power rangers", "superhero team",
            "team", "teammate", "teamwork", "x-men",
        },
        "weight": 0.040,
    },
    "save_world": {
        "query_terms": {"save", "saving", "planet", "world", "earth", "galaxy", "universe"},
        "movie_terms": {
            "alien invasion", "apocalypse", "death star", "earth", "galaxy", "planet",
            "rescue mission", "save the world", "saving the world", "space",
        },
        "weight": 0.040,
    },
    "betrayal": {
        "query_terms": {"betray", "betrays", "betrayed", "betrayal", "traitor", "deception"},
        "movie_terms": {"betrayal", "deception", "double cross", "judas", "traitor"},
        "weight": 0.030,
    },
    "sacrifice": {
        "query_terms": {"sacrifice", "sacrifices", "sacrificed", "sacrificial"},
        "movie_terms": {"human sacrifice", "sacrifice", "sacrificial"},
        "weight": 0.025,
    },
}


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s,.-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def weighted_text(value: object, weight: int) -> str:
    text = normalize_text(value)
    return " ".join([text] * weight)


def significant_tokens(text: str) -> set[str]:
    return {
        token
        for token in normalize_text(text).replace(",", " ").replace(".", " ").split()
        if len(token) > 3 and token not in IMPORTANT_STOPWORDS
    }


def expand_query(query: str) -> str:
    tokens = normalize_text(query).split()
    expanded = [query]
    for token in tokens:
        if token in QUERY_EXPANSIONS:
            expanded.append(QUERY_EXPANSIONS[token])
    return " ".join(expanded)


def requested_genres(query: str) -> set[str]:
    query_tokens = significant_tokens(query)
    query_text = normalize_text(query)
    genres = set()
    for genre, aliases in GENRE_ALIASES.items():
        if query_tokens.intersection(aliases) or genre.lower() in query_text:
            genres.add(genre)
    return genres


def _contains_term(text: str, term: str) -> bool:
    escaped = re.escape(term)
    return re.search(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", text) is not None


@dataclass
class Engine:
    """Holds the fitted vectorizers, sparse matrices and precomputed signals."""

    movies: pd.DataFrame
    word_vectorizer: TfidfVectorizer
    char_vectorizer: TfidfVectorizer
    word_matrix: csr_matrix
    char_matrix: csr_matrix
    token_sets: list[set[str]]
    genre_sets: list[set[str]]
    quality: np.ndarray
    concept_masks: dict[str, np.ndarray]
    action_adventure_mask: np.ndarray
    weak_metadata_penalty: np.ndarray
    team_betrayal_mask: np.ndarray

    # ------------------------------------------------------------------ scoring
    def _concept_intent_score(self, query: str) -> np.ndarray:
        query_text = normalize_text(query)
        query_tokens = significant_tokens(query)
        scores = np.zeros(len(self.movies), dtype=np.float32)

        for name, rule in CONCEPT_RULES.items():
            if query_tokens.intersection(rule["query_terms"]):
                scores += self.concept_masks[name] * float(rule["weight"])

        if "hero" in query_tokens or "heros" in query_tokens or "superhero" in query_text:
            scores += self.action_adventure_mask * 0.025
            scores -= self.weak_metadata_penalty

        if {"team", "betray"}.issubset(query_tokens) or {"team", "betrayed"}.issubset(query_tokens):
            scores += self.team_betrayal_mask * 0.055

        return scores

    def _scores(self, query: str) -> np.ndarray:
        expanded_query = expand_query(query)
        word_query_vector = self.word_vectorizer.transform([normalize_text(expanded_query)])
        word_similarity = cosine_similarity(word_query_vector, self.word_matrix).ravel()

        query_tokens = significant_tokens(expanded_query)
        if query_tokens:
            keyword_overlap = np.array(
                [
                    (
                        len(query_tokens.intersection(movie_tokens))
                        / np.sqrt(len(query_tokens) * len(movie_tokens))
                        if movie_tokens
                        else 0.0
                    )
                    for movie_tokens in self.token_sets
                ],
                dtype=np.float32,
            )
        else:
            keyword_overlap = np.zeros(len(self.movies), dtype=np.float32)

        query_genres = requested_genres(expanded_query)
        if query_genres:
            genre_boost = np.array(
                [
                    len(query_genres.intersection(movie_genres)) / len(query_genres)
                    for movie_genres in self.genre_sets
                ],
                dtype=np.float32,
            )
        else:
            genre_boost = np.zeros(len(self.movies), dtype=np.float32)

        base_scores = (
            (word_similarity * 0.62)
            + (keyword_overlap * 0.10)
            + (genre_boost * 0.08)
            + self._concept_intent_score(query)
            + (self.quality * 0.07)
        )

        candidate_count = min(RERANK_CANDIDATES, len(self.movies))
        candidate_indexes = np.argpartition(base_scores, -candidate_count)[-candidate_count:]
        char_query_vector = self.char_vectorizer.transform([normalize_text(query)])
        candidate_char_similarity = cosine_similarity(
            char_query_vector, self.char_matrix[candidate_indexes]
        ).ravel()

        char_similarity = np.zeros(len(self.movies), dtype=np.float32)
        char_similarity[candidate_indexes] = candidate_char_similarity

        # Broad word TF-IDF retrieval, then typo-tolerant char TF-IDF reranking.
        scores = base_scores + (char_similarity * 0.16)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    # -------------------------------------------------------------- public API
    def recommend(self, query: str, limit: int = TOP_RESULTS) -> list[dict[str, object]]:
        scores = self._scores(query)
        top_indexes = np.argsort(scores)[::-1][:limit]
        return [self._format(self.movies.iloc[idx], float(scores[idx])) for idx in top_indexes]

    @staticmethod
    def _format(row: pd.Series, score: float) -> dict[str, object]:
        return {
            "title": row["title"],
            "overview": row["overview"],
            "genre": row["Genre"] or "Unknown",
            "year": int(row["Release_Year"]) if row["Release_Year"] else "Unknown",
            "language": row["original_language"],
            "rating": round(float(row["vote_average"]), 1),
            "votes": int(row["vote_count"]),
            "poster": row["Poster_Url"],
            "keywords": row["keyword_display"],
            "cast": row["cast_display"],
            "director": row["director_display"],
            "match": round(max(0.0, score) * 100, 1),
        }


def _quality_signal(movies: pd.DataFrame) -> np.ndarray:
    popularity = np.log1p(movies["popularity"].to_numpy(dtype=float))
    votes = np.log1p(movies["vote_count"].to_numpy(dtype=float))
    rating = movies["vote_average"].to_numpy(dtype=float)
    user_rating = movies["user_rating_mean"].to_numpy(dtype=float) / 5
    user_rating_count = np.log1p(movies["user_rating_count"].to_numpy(dtype=float))

    popularity_norm = popularity / popularity.max() if popularity.max() else popularity
    votes_norm = votes / votes.max() if votes.max() else votes
    user_rating_count_norm = (
        user_rating_count / user_rating_count.max() if user_rating_count.max() else user_rating_count
    )
    rating_norm = rating / 10
    tmdb_quality = (rating_norm * 0.55) + (votes_norm * 0.25) + (popularity_norm * 0.20)
    user_quality = (user_rating * 0.70) + (user_rating_count_norm * 0.30)
    return ((tmdb_quality * 0.72) + (user_quality * 0.28)).astype(np.float32)


def build_engine(data_path: str | Path = DEFAULT_DATA_PATH) -> Engine:
    """Load the processed catalogue and fit the TF-IDF model. Called once and cached."""
    data_path = Path(data_path)
    movies = pd.read_csv(data_path)

    text_columns = [
        "overview", "tagline", "Genre", "keyword_text", "keyword_display",
        "cast_text", "cast_display", "director_text", "director_display",
        "original_language", "Poster_Url", "title",
    ]
    for column in text_columns:
        if column in movies.columns:
            movies[column] = movies[column].fillna("")
    movies["Release_Year"] = pd.to_numeric(movies["Release_Year"], errors="coerce").fillna(0).astype(int)
    for column in ["popularity", "vote_count", "vote_average", "user_rating_mean", "user_rating_count"]:
        movies[column] = pd.to_numeric(movies[column], errors="coerce").fillna(0)

    movies["search_text"] = (
        movies["title"].map(lambda v: weighted_text(v, 3)) + " "
        + movies["Genre"].map(lambda v: weighted_text(str(v).replace(",", " "), 6)) + " "
        + movies["keyword_text"].map(lambda v: weighted_text(v, 8)) + " "
        + movies["tagline"].map(lambda v: weighted_text(v, 3)) + " "
        + movies["overview"].map(lambda v: weighted_text(v, 4)) + " "
        + movies["cast_text"].map(lambda v: weighted_text(v, 2)) + " "
        + movies["director_text"].map(lambda v: weighted_text(v, 2))
    )
    token_sets = movies["search_text"].map(significant_tokens).tolist()
    genre_sets = movies["Genre"].map(
        lambda v: set(str(v).split(", ")) if v else set()
    ).tolist()

    word_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_matrix = word_vectorizer.fit_transform(movies["search_text"])
    char_matrix = char_vectorizer.fit_transform(movies["search_text"])

    # Precompute the corpus-side concept signals once (query-independent).
    intent_text = (
        movies["title"] + " " + movies["Genre"] + " " + movies["keyword_text"] + " "
        + movies["tagline"] + " " + movies["overview"]
    ).map(normalize_text).tolist()

    concept_masks: dict[str, np.ndarray] = {}
    for name, rule in CONCEPT_RULES.items():
        terms = rule["movie_terms"]
        concept_masks[name] = np.array(
            [any(_contains_term(text, term) for term in terms) for text in intent_text],
            dtype=np.float32,
        )

    action_adventure_mask = np.array(
        [
            float("Action" in g and ("Adventure" in g or "Science Fiction" in g))
            for g in genre_sets
        ],
        dtype=np.float32,
    )
    weak_metadata_penalty = (
        movies["keyword_text"].str.len().to_numpy() == 0
    ).astype(np.float32) * 0.025
    team_betrayal_mask = np.array(
        [
            float(
                ("superhero team" in text or "team" in text)
                and ("betrayal" in text or "judas" in text or "traitor" in text)
            )
            for text in intent_text
        ],
        dtype=np.float32,
    )

    movies = movies.drop(columns=["search_text"]).reset_index(drop=True)

    return Engine(
        movies=movies,
        word_vectorizer=word_vectorizer,
        char_vectorizer=char_vectorizer,
        word_matrix=word_matrix,
        char_matrix=char_matrix,
        token_sets=token_sets,
        genre_sets=genre_sets,
        quality=_quality_signal(movies),
        concept_masks=concept_masks,
        action_adventure_mask=action_adventure_mask,
        weak_metadata_penalty=weak_metadata_penalty,
        team_betrayal_mask=team_betrayal_mask,
    )


if __name__ == "__main__":
    import sys

    engine = build_engine()
    sample = " ".join(sys.argv[1:]) or (
        "I want to see a hero movie where the hero sacrifices themself to save a "
        "planet but the team betrays the hero."
    )
    print(f"Query: {sample}\n")
    for rank, movie in enumerate(engine.recommend(sample, limit=10), start=1):
        print(f"{rank:2d}. {movie['title']} ({movie['year']})  "
              f"[{movie['genre']}]  match={movie['match']}%")
