# EDA Results

Generated from the deployed artifacts:

- `data/movies.csv.gz`
- `data/cf_neighbors.csv.gz`

## Dataset Summary

| Metric | Result |
|---|---:|
| Movies | 45,197 |
| Release years | 1874-2020 |
| Poster coverage | 99.2% |
| Collaborative-filtering covered movies | 19,779 |
| Collaborative-filtering coverage rate | 43.8% |

## Text-Feature Coverage

These are the fields the TF-IDF engine ranks on, so their coverage bounds how well any
given movie can be matched. Overview, cast, and director are near-complete; tagline and
keywords are sparser and lean more on the other signals.

| Field | Coverage | Mean words |
|---|---:|---:|
| Overview | 98.4% | 55.1 |
| Tagline | 45.1% | 8.5 |
| Keywords | 68.8% | 7.4 |
| Cast | 94.8% | 14.3 |
| Director | 98.1% | 2.3 |

## MovieLens User-Rating Coverage

The MovieLens user-rating signal (a tie-breaker in ranking) is present for **20.0%**
(9,024 of 45,197) of movies. The remaining titles fall back to TMDB rating/vote/popularity
only.

## Movies per Decade

Catalogue is heavily weighted toward 2000s–2010s releases (~53% of all titles).

| Decade | Count |
|---|---:|
| 1870s | 2 |
| 1880s | 4 |
| 1890s | 75 |
| 1900s | 87 |
| 1910s | 176 |
| 1920s | 431 |
| 1930s | 1,315 |
| 1940s | 1,491 |
| 1950s | 2,069 |
| 1960s | 2,603 |
| 1970s | 3,448 |
| 1980s | 3,904 |
| 1990s | 5,636 |
| 2000s | 11,133 |
| 2010s | 12,748 |
| 2020s | 1 |

## Top Genres

| Genre | Count |
|---|---:|
| Drama | 20,243 |
| Comedy | 13,176 |
| Thriller | 7,618 |
| Romance | 6,730 |
| Action | 6,590 |
| Horror | 4,670 |
| Crime | 4,304 |
| Documentary | 3,930 |
| Adventure | 3,490 |
| Science Fiction | 3,042 |

## Top Original Languages

| Language | Count |
|---|---:|
| EN | 32,231 |
| FR | 2,412 |
| IT | 1,449 |
| JA | 1,345 |
| DE | 1,062 |
| ES | 979 |
| RU | 814 |
| HI | 508 |
| KO | 444 |
| ZH | 408 |

## Top-Rated Movies

Minimum `1,000` TMDB votes.

| Title | Year | Genre | TMDB Rating | Vote Count |
|---|---:|---|---:|---:|
| The Shawshank Redemption | 1994 | Drama, Crime | 8.5 | 8,358 |
| The Godfather | 1972 | Drama, Crime | 8.5 | 6,024 |
| Your Name. | 2016 | Romance, Animation, Drama | 8.5 | 1,030 |
| The Dark Knight | 2008 | Drama, Action, Crime, Thriller | 8.3 | 12,269 |
| Fight Club | 1999 | Drama | 8.3 | 9,678 |
| Pulp Fiction | 1994 | Thriller, Crime | 8.3 | 8,670 |
| Schindler's List | 1993 | Drama, History, War | 8.3 | 4,436 |
| Whiplash | 2014 | Drama | 8.3 | 4,376 |
| Spirited Away | 2001 | Fantasy, Adventure, Animation, Family | 8.3 | 3,968 |
| Life Is Beautiful | 1997 | Comedy, Drama | 8.3 | 3,643 |

Reproduce this report with:

```bash
python3 eda.py
```
