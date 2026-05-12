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
|-- main.ipynb
|-- requirements.txt
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

The trained model files from the local `models/` directory are intentionally excluded because they are several gigabytes each. Re-run the training cells in `main.ipynb` to regenerate them.

## Setup

```bash
git lfs install
git lfs pull
python -m pip install -r requirements.txt
jupyter notebook main.ipynb
```

## Main Notebook

`main.ipynb` is organized into:

- Data loading and inspection.
- Metadata preprocessing and TF-IDF feature generation.
- SVD baseline evaluation.
- Hybrid meta-training dataset usage.
- XGBoost, LightGBM, and CatBoost comparison.
- Similar-anime recommendation examples.
- Optional visualization and interpretation sections.
