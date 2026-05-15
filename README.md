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
It also creates `models/svd_light.pkl`, which avoids loading the full Surprise trainset during inference.

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

`recommend-user` is the main hybrid recommender. It uses SVD for CF scores and the meta learner for final ranking. When optimized files are present, it prefers `svd_light.pkl`, `meta_model_core.pkl`, and `item_sim_matrix.float32.npy` to reduce memory use.

Create an interactive 3D anime embedding map:

```bash
python main.py visualize-3d --output artifacts/anime_tsne_3d.html
python main.py visualize-3d --user-id 116169 --output artifacts/user_116169_tsne_3d.html
```

## Web App

Run the web version:

```bash
python app.py --host 0.0.0.0 --port 8000
```

Open `http://SERVER_IP:8000`. Users can search and select anime titles, then get content-based recommendations directly on the page. The web UI defaults to avoiding repeated entries from the same franchise so recommendations feel more discovery-oriented.

The web app also exposes the generated interactive 3D anime map at `/map`. Create it first with:

```bash
python main.py visualize-3d --output artifacts/anime_tsne_3d.html
```

For a production-style process:

```bash
gunicorn -w 1 -b 0.0.0.0:8000 'anime_recommender.web:create_app()'
```

To keep the site alive after disconnecting VS Code or SSH, run it in `tmux`:

```bash
tmux new-session -d -s anime_web 'cd ~/animation_recommendation && gunicorn -w 1 -b 0.0.0.0:8000 "anime_recommender.web:create_app()"'
```

If `http://SERVER_IP:8000` does not open from another device, the server is listening correctly but the public network/firewall is blocking port 8000. In that case, either open TCP 8000 or use a tunnel service such as Cloudflare Tunnel or ngrok.

## Member Hybrid Web App

Run the separate member-based hybrid version:

```bash
python member_app.py --host 0.0.0.0 --port 8100
```

Users can sign up, rate anime from search results, and unlock personalized recommendations after 10 ratings. New users are handled with an SVD fold-in vector plus content-based profile scores, so ratings can affect recommendations immediately without retraining the full SVD model on every click. Full SVD retraining should be done later as a batch job after enough new ratings accumulate.

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
