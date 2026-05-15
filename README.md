# Anime Recommendation System

Hybrid anime recommendation project combining collaborative filtering, content-based metadata features, and gradient-boosting meta learners.

## Project Summary

The project builds an anime recommender with three main signals:

- Collaborative filtering with Surprise SVD on user-anime ratings.
- Content-based features from anime metadata such as genres, studios, producers, type, source, rating, and popularity.
- Meta learners such as LightGBM, XGBoost, and CatBoost that learn how to combine CF and CB scores.

The report and presentation were used as project references. They are not committed because this repository focuses on the reproducible code and data pipeline.

## Repository Structure

```text
.
|-- main.py
|-- main.ipynb
|-- requirements.txt
|-- anime_recommender/
|   |-- inference.py
|   |-- training.py
|   |-- preprocessing.py
|   |-- config.py
|   `-- io.py
|-- artifacts/
|   `-- tfidf_and_encoders.pkl
|-- data/
|   |-- raw/
|   |   |-- anime.csv
|   |   |-- anime_test.csv
|   |   |-- rating_complete.csv
|   |   `-- rating_test.csv
|   `-- processed/
|       |-- meta_preprocessed.csv
|       `-- meta_train_ready.csv
`-- models/
    `-- .gitkeep
```

## Data Notes

`rating_complete.csv` and `rating_test.csv` are larger than GitHub's normal 100MB file limit, so they are tracked with Git LFS.

The trained model files from the local `models/` directory are intentionally excluded because they are several gigabytes each. If you already have them locally, put them here:

```text
models/svd_model.pkl
models/meta_model.pkl
```

Normal hybrid inference uses `svd_model.pkl` and `meta_model.pkl`. The current `meta_model.pkl` package already contains the content-based `item_sim_matrix`, so `models/item_sim_matrix_all.pkl` is only an optional fallback/debug artifact.

For memory-friendly server inference, run:

```bash
python main.py optimize-models
```

This creates `models/meta_model_core.pkl` and `models/item_sim_matrix.float32.npy`. The CLI will automatically prefer these optimized files when present.

## Setup

```bash
git lfs install
git lfs pull
python -m pip install -r requirements.txt
```

If you already have the large CSV files locally, you can copy them directly into `data/raw/` instead of using Git LFS.

## CLI Usage

The primary entry point is `main.py`.

Check whether data and models are in the expected paths:

```bash
python main.py check
```

Run inference with existing model files:

```bash
python main.py similar --title "Naruto" --top-n 10
python main.py recommend-user --user-id 116169 --top-n 10
python main.py meta-similar --title "Koe no Katachi" --top-n 10
```

`similar` is the lightweight content-similarity command. It does not load the multi-GB model files, so it is the safest first smoke test on a server.

`recommend-user` is the main hybrid recommender. It loads `svd_model.pkl` for CF scores and `meta_model.pkl` for the LightGBM-style meta learner plus the content-based similarity matrix.

Train only the meta learner from the precomputed `meta_train_ready.csv`:

```bash
python main.py train-meta
```

Run the full heavy training pipeline from raw ratings:

```bash
python main.py train-full --sample-users 3000
```

## Notebook

`main.ipynb` is kept as an exploratory/report notebook. For server runs, prefer the CLI pipeline above because training and inference are separated cleanly.
