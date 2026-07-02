# bir·mov

Personal movie recommendations from your own IMDb or Letterboxd ratings, powered by
two collaborative-filtering models trained on MovieLens ml-32m (see
`ml-32m-train.ipynb`):

- [`mskayacioglu/ml32m-als128-v1`](https://huggingface.co/mskayacioglu/ml32m-als128-v1) —
  implicit-feedback ALS, does the **ranking** (what to watch next)
- [`mskayacioglu/ml32m-mf128-v1`](https://huggingface.co/mskayacioglu/ml32m-mf128-v1) —
  explicit matrix factorization, does the **scoring** (the rating you'd likely give)

Neither model has ever seen you, so your uploaded ratings are folded in at request
time: ALS via `recalculate_user`, MF via a closed-form ridge solve against the
frozen item factors.

![theme](web/bir_mark.png)

## What it does

1. Upload an **IMDb ratings CSV**, a **Letterboxd `ratings.csv`**, or a full
   **Letterboxd export `.zip`** (drag & drop).
2. Titles are matched to the MovieLens catalog — IMDb exports by `tt` id via
   `links.csv`, Letterboxd by normalized title + year (±1 tolerance).
3. You get a ranked list with, per movie:
   - predicted score on **IMDb's 10-point** and **Letterboxd's 5-star** scale
   - match strength, confidence, and *"because you liked …"* explanations
   - genre chips, community average + vote count
   - direct **IMDb** and **Letterboxd** links
4. Export the list as **CSV**, **JSON**, or a **Letterboxd-import CSV**.

The UI reuses the [bir](../bir) dashboard theme.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# MovieLens data (movies.csv/links.csv are required at runtime, ratings.csv
# only for the one-time stats build): https://grouplens.org/datasets/movielens/32m/
# unzip into ./ml-32m

# one-time: aggregate ratings.csv into data/movie_stats.csv
.venv/bin/python scripts/build_movie_stats.py
```

Model artifacts are picked up from `model/` when present, otherwise downloaded
from Hugging Face into `.hf-cache/` on first start.

## Run

```bash
.venv/bin/uvicorn app.main:app --port 8360
```

Open <http://127.0.0.1:8360>.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## Layout

```
app/            FastAPI backend
  catalog.py    movies/links/stats + title-year matching
  parsers.py    IMDb / Letterboxd export parsing
  recommender.py  model loading + fold-in ranking & scoring
  main.py       API + static hosting
web/            static frontend (bir theme)
scripts/        one-time data preparation
tests/          parser + catalog tests
```
