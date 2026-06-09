"""Refresh stale TMDB poster URLs in ``data/movies.csv.gz`` from the live TMDB API.

The Kaggle "The Movies Dataset" is a 2017 snapshot, so ~75% of its ``poster_path`` values
now 404 on TMDB's CDN — the underlying image files were deleted or replaced. This script
re-queries the TMDB API by movie ``id``, writes the *current* poster path back into the
catalogue, and rebuilds ``data/movies.csv.gz`` so the app shows live posters again.

Get a free TMDB credential at https://www.themoviedb.org/settings/api — either a v3
"API Key" or a v4 "API Read Access Token" works.

Usage
-----
    export TMDB_API_KEY=your_v3_key        # OR
    export TMDB_BEARER=your_v4_read_token

    python refresh_posters.py              # refresh every movie (resumable)
    python refresh_posters.py --limit 50   # smoke-test on the first 50 movies
    python refresh_posters.py --workers 16 # tune concurrency

Progress is cached to ``data/poster_cache.csv`` after every batch, so you can stop and
re-run without losing work — already-fetched movies are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CATALOGUE_PATH = BASE_DIR / "data" / "movies.csv.gz"
CACHE_PATH = BASE_DIR / "data" / "poster_cache.csv"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie/{id}"


def _credentials(args: argparse.Namespace) -> tuple[str | None, str | None]:
    api_key = args.api_key or os.environ.get("TMDB_API_KEY")
    bearer = args.bearer or os.environ.get("TMDB_BEARER")
    if not api_key and not bearer:
        raise SystemExit(
            "No TMDB credential found. Set TMDB_API_KEY (v3) or TMDB_BEARER (v4), or pass "
            "--api-key / --bearer. Get one free at https://www.themoviedb.org/settings/api"
        )
    return api_key, bearer


def fetch_poster_path(movie_id: int, api_key: str | None, bearer: str | None) -> str:
    """Return the current poster_path for a movie ('' if none / movie removed)."""
    url = TMDB_MOVIE_URL.format(id=movie_id)
    headers = {"Accept": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"
    else:
        url += f"?api_key={api_key}"

    for attempt in range(5):
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("poster_path") or ""
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return ""  # movie no longer exists on TMDB
            if error.code == 401:
                raise SystemExit("TMDB rejected the credential (401). Check your key/token.")
            if error.code == 429:  # rate limited — honour Retry-After, then retry
                wait = int(error.headers.get("Retry-After", "2"))
                time.sleep(wait + 1)
                continue
            time.sleep(1 + attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1 + attempt)
    return ""  # gave up after retries; leave poster blank


def load_cache() -> dict[int, str]:
    if not CACHE_PATH.exists():
        return {}
    cached = pd.read_csv(CACHE_PATH).fillna({"poster_path": ""})
    return {int(r.id): str(r.poster_path) for r in cached.itertuples()}


def save_cache(cache: dict[int, str]) -> None:
    frame = pd.DataFrame(
        {"id": list(cache.keys()), "poster_path": list(cache.values())}
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE_PATH, index=False)


def refresh(args: argparse.Namespace) -> None:
    api_key, bearer = _credentials(args)
    movies = pd.read_csv(CATALOGUE_PATH)
    ids = movies["id"].astype(int).tolist()
    if args.limit:
        ids = ids[: args.limit]

    cache = load_cache()
    todo = [movie_id for movie_id in ids if movie_id not in cache]
    print(f"{len(ids):,} movies in scope · {len(cache):,} already cached · {len(todo):,} to fetch")
    if not todo:
        print("Nothing to fetch — rebuilding catalogue from cache.")

    lock = threading.Lock()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_poster_path, movie_id, api_key, bearer): movie_id
            for movie_id in todo
        }
        for future in as_completed(futures):
            movie_id = futures[future]
            with lock:
                cache[movie_id] = future.result()
                done += 1
                if done % 200 == 0 or done == len(todo):
                    save_cache(cache)
                    print(f"  fetched {done:,}/{len(todo):,}", end="\r")
    print()
    save_cache(cache)

    # Rebuild Poster_Url from the refreshed paths.
    new_urls = movies["id"].astype(int).map(
        lambda mid: f"{TMDB_IMAGE_BASE_URL}{cache[mid]}" if cache.get(mid) else ""
    )
    before = movies["Poster_Url"].fillna("").str.strip().ne("").mean()
    movies["Poster_Url"] = new_urls
    after = new_urls.str.strip().ne("").mean()

    CATALOGUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    movies.to_csv(CATALOGUE_PATH, index=False, compression="gzip")
    print(
        f"Rewrote {CATALOGUE_PATH} — poster URLs present: {before:.1%} -> {after:.1%} "
        f"({new_urls.str.strip().ne('').sum():,} of {len(movies):,} movies)"
    )
    print("Note: 'present' means a URL exists; the refreshed paths now point at live "
          "TMDB files, so they should actually load.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key", help="TMDB v3 API key (or set TMDB_API_KEY).")
    parser.add_argument("--bearer", help="TMDB v4 read access token (or set TMDB_BEARER).")
    parser.add_argument("--workers", type=int, default=16, help="Concurrent requests.")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N movies (testing).")
    refresh(parser.parse_args())


if __name__ == "__main__":
    main()
