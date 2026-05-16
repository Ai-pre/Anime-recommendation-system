# Anime Recommendation System 정리본

## 1. 프로젝트 한 줄 요약

애니메이션 메타데이터와 유저 평점 데이터를 함께 사용하여, `SVD 기반 협업 필터링(CF)`과 `콘텐츠 기반 추천(CB)`을 결합한 하이브리드 애니메이션 추천 시스템을 구현했다. 최종 추천 점수는 `cf_score`와 `cb_score`를 입력으로 받는 `LightGBM Meta-Learner`가 예측한다.

## 2. 프로젝트 목표

기존 단일 추천 방식의 한계를 줄이고, 유저의 실제 평점 패턴과 애니메이션의 장르/제작사/스튜디오/인기도 정보를 함께 반영하는 추천 시스템을 만드는 것이 목표였다.

- 협업 필터링은 유저와 아이템 사이의 평점 패턴을 학습한다.
- 콘텐츠 기반 추천은 장르, 제작사, 스튜디오, 타입, 점수, 인기도 등 애니 자체의 특징을 사용한다.
- 메타러너는 CF와 CB가 만든 두 점수를 다시 학습하여 최종 예측 평점을 만든다.
- 최종 결과는 CLI, 콘텐츠 기반 웹 데모, 회원제 하이브리드 웹 데모, 3D 애니 라이브러리로 확인할 수 있다.

## 3. 사용 데이터

### 3.1 Raw Data

| 파일 | 역할 |
| --- | --- |
| `data/raw/anime.csv` | 애니메이션 메타데이터. 제목, 장르, 타입, 제작사, 스튜디오, 점수, 인기도, 멤버 수 등을 포함한다. |
| `data/raw/rating_complete.csv` | 학습용 유저-애니 평점 데이터. `user_id`, `anime_id`, `rating` 형태로 구성된다. |
| `data/raw/rating_test.csv` | SVD, NeuMF 같은 CF 모델의 외부 테스트에 사용할 수 있는 평점 데이터. |
| `data/raw/anime_test.csv` | 테스트용 애니 메타데이터였지만, 학습용 `anime.csv`와 컬럼 구조가 달라서 최종 하이브리드 평가에서는 제외했다. |

보고서 기준으로 `anime.csv`와 `rating_complete.csv`는 결측치나 0점 unrated 값이 거의 없어 추천 모델 학습에 적합했다. 다만 `anime_test.csv`는 학습 데이터와 동일한 TF-IDF/메타데이터 feature space를 재현하기 어려워, CB 점수 계산과 하이브리드 테스트에는 사용하지 않았다.

### 3.2 Processed Data

| 파일 | 역할 |
| --- | --- |
| `data/processed/meta_preprocessed.csv` | 애니별 콘텐츠 기반 feature vector. TF-IDF, 인코딩, 정규화된 수치 변수가 합쳐진다. |
| `data/processed/meta_train_ready.csv` | 메타러너 학습용 데이터. `user_id`, `anime_id`, `cf_score`, `cb_score`, `true_rating`으로 구성된다. |
| `artifacts/tfidf_and_encoders.pkl` | TF-IDF vectorizer, LabelEncoder, scaler를 저장한 파일. 학습과 추론의 전처리 일관성을 보장한다. |

## 4. 사용 기술과 라이브러리

### 4.1 데이터 처리

- `pandas`: CSV 로딩, 전처리, join, groupby, meta-train dataset 생성
- `numpy`: 행렬 연산, cosine similarity, SVD factor 연산, score clipping
- `scikit-learn`: TF-IDF, LabelEncoder, MinMaxScaler, StandardScaler, train-test split, RMSE 계산, PCA, t-SNE

### 4.2 추천 모델

- `scikit-surprise`: SVD 협업 필터링 모델 학습
- `LightGBM`: 최종 메타러너. 최종 선택 모델
- `XGBoost`: 비교용 메타러너
- `CatBoost`: 비교용 메타러너
- `TensorFlow/Keras`: 보고서 단계에서 NeuMF 실험에 사용

### 4.3 시각화와 서비스

- `plotly`: 3D 애니 라이브러리 시각화
- `Flask`: 웹 API와 화면 렌더링
- `SQLite`: 회원, 평점, 조회수 저장
- `gunicorn`: 서버 실행
- `Cloudflare Tunnel`: 외부 공유용 임시 공개 URL

## 5. 전체 학습 파이프라인

```text
anime.csv + rating_complete.csv
        |
        v
메타데이터 전처리
        |
        v
meta_preprocessed.csv 생성
        |
        +----------------------+
        |                      |
        v                      v
Surprise SVD 학습        콘텐츠 기반 유사도 계산
        |                      |
        v                      v
cf_score 생성            cb_score 생성
        |                      |
        +----------+-----------+
                   |
                   v
meta_train_ready.csv 생성
                   |
                   v
LightGBM / XGBoost / CatBoost 학습
                   |
                   v
가장 성능이 좋은 Meta-Learner 선택
                   |
                   v
최종 하이브리드 추천
```

## 6. 전처리 과정

### 6.1 메타데이터 로딩과 결측치 처리

`anime.csv`를 불러온 뒤, 애니메이션 ID인 `MAL_ID`를 정수형으로 통일했다. 장르, 제작사, 스튜디오처럼 텍스트 기반 feature는 결측치를 `Unknown` 또는 빈 문자열로 처리하고, `Type`, `Source`, `Rating`, `Premiered`, `Duration` 같은 범주형 feature도 결측치가 있으면 `Unknown`으로 채웠다.

이 과정은 TF-IDF와 LabelEncoder가 결측치 때문에 실패하지 않게 만드는 단계다.

### 6.2 범주형 feature 인코딩

다음 컬럼은 `LabelEncoder`로 숫자화했다.

- `Type`
- `Source`
- `Rating`
- `Premiered`
- `Duration`

범주형 문자열을 숫자로 바꿔야 트리 모델과 유사도 계산에 사용할 수 있다. 학습 당시 사용한 encoder는 `tfidf_and_encoders.pkl`에 저장해서 추론 시에도 같은 매핑을 유지한다.

### 6.3 TF-IDF feature 생성

다음 텍스트 컬럼은 TF-IDF로 벡터화했다.

- `Genres`
- `Producers`
- `Studios`

각 컬럼별 최대 feature 수는 100개로 제한했다. 장르, 제작사, 스튜디오 정보는 애니의 콘텐츠적 특징을 가장 잘 나타내는 정보이기 때문에 콘텐츠 기반 추천의 핵심 feature로 사용했다.

### 6.4 수치형 feature 정규화

보고서와 최종 모델에서 핵심적으로 사용한 수치형 feature는 다음과 같다.

- `Score`
- `Episodes`
- `Ranked`
- `Popularity`
- `Members`
- `Favorites`

수치형 feature는 스케일 차이가 크다. 예를 들어 `Members`는 수백만 단위일 수 있지만 `Score`는 1~10 범위다. 따라서 `MinMaxScaler`로 0~1 범위로 정규화하여 특정 feature가 유사도 계산을 과도하게 지배하지 않도록 했다.

### 6.5 콘텐츠 기반 feature table 생성

최종적으로 아래 feature들을 하나로 합쳐 `meta_preprocessed.csv`를 만들었다.

- `MAL_ID`
- 정규화된 수치형 feature
- 인코딩된 범주형 feature
- `Genres` TF-IDF feature
- `Producers` TF-IDF feature
- `Studios` TF-IDF feature

이 파일은 애니 하나당 하나의 콘텐츠 벡터를 가지며, 이후 cosine similarity와 메타러너 학습에 사용된다.

## 7. 협업 필터링 모델: SVD

### 7.1 SVD를 사용한 이유

SVD는 유저-아이템 평점 행렬을 저차원 latent factor로 분해하여, 유저가 아직 보지 않은 애니에 줄 평점을 예측한다. 애니 추천에서는 유저가 실제로 남긴 평점 패턴이 중요하기 때문에 CF 모델로 SVD를 사용했다.

### 7.2 주요 설정

| 파라미터 | 값 | 의미 |
| --- | --- | --- |
| `n_factors` | 100 | 유저와 아이템을 표현할 latent vector 차원 |
| `n_epochs` | 20 | 학습 반복 횟수 |
| `lr_all` | 0.005 | 학습률 |
| `reg_all` | 0.02 | 과적합 방지를 위한 정규화 |
| `random_state` | 42 | 재현성을 위한 seed |

SVD의 출력은 다음과 같이 사용된다.

```text
cf_score = svd.predict(user_id, anime_id).est
```

여기서 `cf_score`는 특정 유저가 특정 애니에 줄 것으로 예상되는 협업 필터링 기반 평점이다.

## 8. NeuMF 실험

보고서에서는 SVD의 선형적 한계를 보완하기 위해 NeuMF도 실험했다. NeuMF는 `GMF`와 `MLP`를 결합해 유저와 아이템의 비선형 상호작용을 학습하는 Neural Collaborative Filtering 계열 모델이다.

다만 최종 레포와 서비스에서는 NeuMF를 사용하지 않았다.

- 평점 데이터가 8~9점대에 많이 몰려 있어 class imbalance가 있었다.
- 파라미터 수가 많아 validation에서 과적합 문제가 발생했다.
- 서버 배포와 추론 안정성 측면에서 SVD가 더 단순하고 안정적이었다.

따라서 최종 하이브리드 시스템의 CF 모델은 Surprise SVD로 고정했다.

## 9. 콘텐츠 기반 추천: CB

### 9.1 아이템 간 유사도

`meta_preprocessed.csv`의 feature vector를 `StandardScaler`로 표준화한 뒤 cosine similarity를 계산했다.

```text
item_sim_matrix = cosine_similarity(standardized_meta_features)
```

이 행렬은 애니와 애니 사이의 콘텐츠 유사도를 나타낸다. 장르, 제작사, 스튜디오, 타입, 점수, 인기도 등이 비슷한 작품일수록 유사도가 높아진다.

### 9.2 유저 콘텐츠 프로필

특정 유저의 콘텐츠 취향을 만들 때는, 유저가 높게 평가한 애니만 사용한다.

- 기본 좋아요 기준: `rating >= 8.0`
- 각 liked anime마다 가장 유사한 `top_k=30`개 이웃만 유지
- 여러 liked anime의 유사도 벡터를 평균내서 유저의 콘텐츠 기반 선호 벡터 생성

이렇게 만든 점수가 `cb_score`다.

```text
cb_score = user_content_profile[anime_id]
```

Top-K neighbor masking을 적용한 이유는 모든 애니의 유사도를 평균내면 노이즈가 커지기 때문이다. 가장 가까운 이웃만 남기면 콘텐츠 취향이 더 선명해진다.

## 10. 메타러너 학습 데이터 생성

메타러너는 CF와 CB 점수를 합치는 모델이다. 이를 학습하기 위해 `meta_train_ready.csv`를 만들었다.

각 row는 다음 구조를 가진다.

| 컬럼 | 의미 |
| --- | --- |
| `user_id` | 유저 ID |
| `anime_id` | 애니 ID |
| `cf_score` | SVD가 예측한 평점 |
| `cb_score` | 콘텐츠 기반 선호 점수 |
| `true_rating` | 실제 유저 평점 |

생성 과정은 다음과 같다.

1. `rating_complete.csv`에서 최대 3000명의 유저를 샘플링한다.
2. 각 유저의 liked anime를 기준으로 CB profile을 계산한다.
3. 해당 유저가 실제로 평가한 애니마다 SVD로 `cf_score`를 만든다.
4. 같은 애니의 콘텐츠 기반 점수인 `cb_score`를 가져온다.
5. 실제 평점 `true_rating`과 함께 저장한다.

이 데이터셋을 사용하면 메타러너는 다음 관계를 학습한다.

```text
final_score = f(cf_score, cb_score)
```

즉, 단순히 CF와 CB를 고정 비율로 섞는 것이 아니라, 데이터에서 어떤 상황에 CF를 더 믿고 어떤 상황에 CB를 더 믿을지 학습하게 된다.

## 11. 메타러너: LightGBM / XGBoost / CatBoost

### 11.1 비교한 모델

| 모델 | 특징 | 프로젝트 내 역할 |
| --- | --- | --- |
| LightGBM | leaf-wise histogram boosting, 빠른 학습, 낮은 메모리 사용 | 최종 선택 모델 |
| XGBoost | level-wise boosting, 강한 정규화와 안정성 | 비교 모델 |
| CatBoost | categorical feature에 강하고 과적합 방지에 유리 | 비교 모델 |

### 11.2 최종 선택: LightGBM

보고서와 최종 모델 패키지 기준으로 LightGBM이 최종 메타러너로 선택되었다.

최종 서버 모델 정보:

| 항목 | 값 |
| --- | --- |
| 최종 메타러너 | LightGBM |
| 검증 RMSE | 0.7734500591 |
| 학습 샘플 유저 수 | 3000 |
| 전체 유저 수 | 310059 |
| liked 기준 | rating >= 8.0 |
| CB top-k neighbors | 30 |
| 처리된 애니 수 | 16872 |

LightGBM을 선택한 이유는 다음과 같다.

- 입력 feature가 `cf_score`, `cb_score`처럼 작고 연속적인 값이라 LightGBM이 효율적으로 학습할 수 있다.
- leaf-wise growth 방식이 CF와 CB 사이의 비선형 관계를 잘 포착한다.
- XGBoost보다 빠르고, CatBoost보다 현재 feature 구조에 더 잘 맞았다.
- 최종 RMSE와 ranking metric에서 가장 좋은 결과를 보였다.

## 12. 평가 방법

### 12.1 RMSE

RMSE는 예측 평점과 실제 평점의 차이를 측정한다.

```text
RMSE = sqrt(mean((true_rating - predicted_rating)^2))
```

낮을수록 실제 평점에 가깝게 예측했다는 뜻이다. 보고서에서는 raw prediction을 기준으로 RMSE를 계산했다.

### 12.2 Precision@10 / Recall@10

Top-K 추천 품질을 평가하기 위해 Precision@10과 Recall@10도 사용했다.

- `Precision@10`: 추천한 10개 중 실제로 유저가 좋아한 애니의 비율
- `Recall@10`: 유저가 좋아한 전체 애니 중 추천 Top 10 안에 들어온 비율
- liked 기준은 `rating >= 8.0`

평점 예측만 잘하는 모델이 실제 추천 순위까지 좋은 것은 아니기 때문에, RMSE와 Top-K 지표를 함께 사용했다.

## 13. 최종 추론 방식

### 13.1 기존 유저 추천

기존 `rating_complete.csv`에 존재하는 유저는 다음 방식으로 추천한다.

1. 유저의 과거 평점을 불러온다.
2. 이미 본 애니는 후보에서 제거한다.
3. SVD로 후보 애니의 `cf_score`를 예측한다.
4. 유저가 좋아한 애니를 기준으로 CB profile을 만들고 `cb_score`를 계산한다.
5. `LightGBM Meta-Learner`에 `[cf_score, cb_score]`를 넣어 최종 예측 평점을 만든다.
6. 최종 점수가 높은 순서대로 Top-N 애니를 추천한다.

### 13.2 신규 회원 추천

회원제 웹 버전에서는 새 유저가 들어와도 바로 추천이 가능하도록 구현했다.

- 회원가입 후 애니를 검색해서 1~10점으로 평가한다.
- 평가 수가 10개 미만이면 개인화 추천 대신 콘텐츠 기반 검색/추천을 중심으로 보여준다.
- 평가 수가 10개 이상이면 개인화 하이브리드 추천을 보여준다.

신규 회원은 기존 SVD 모델에 존재하지 않는 유저이므로, 전체 SVD를 매번 재학습하지 않는다. 대신 `svd_light.pkl`의 item factor를 활용해 사용자의 임시 latent vector를 fold-in 방식으로 계산한다. 이 방식은 서버에서 빠르게 개인 추천을 만들 수 있고, 실제 서비스에서는 일정량의 신규 평점이 쌓이면 batch retraining으로 SVD를 다시 학습하는 구조가 적합하다.

## 14. 모델 저장 구조

### 14.1 원본 모델 파일

| 파일 | 역할 |
| --- | --- |
| `models/svd_model.pkl` | Surprise SVD 원본 모델 |
| `models/meta_model.pkl` | LightGBM 메타러너, ID mapping, item similarity matrix를 포함한 원본 패키지 |
| `models/item_sim_matrix_all.pkl` | 전체 item-item similarity matrix 백업/디버그용 |

원본 파일은 수 GB 크기라 서버 메모리를 많이 사용한다.

### 14.2 배포 최적화 파일

| 파일 | 역할 |
| --- | --- |
| `models/svd_light.pkl` | Surprise trainset 전체 대신 예측에 필요한 factor/bias만 저장한 경량 SVD |
| `models/meta_model_core.pkl` | item similarity matrix를 제거한 경량 LightGBM 패키지 |
| `models/item_sim_matrix.float32.npy` | matrix를 float32 `.npy`로 저장하고 memory-map 방식으로 읽기 위한 파일 |

최종 서버에서는 메모리를 줄이기 위해 이 최적화 파일들을 우선 사용한다.

## 15. 실행 명령어

### 15.1 파일 확인

```bash
python main.py check
```

### 15.2 콘텐츠 기반 유사 애니 추천

```bash
python main.py similar --title "Naruto" --top-n 10
```

이 명령은 큰 모델 파일을 로드하지 않고 `meta_preprocessed.csv`만 사용하므로 서버 smoke test에 적합하다.

### 15.3 기존 유저 하이브리드 추천

```bash
python main.py recommend-user --user-id 116169 --top-n 10
```

### 15.4 메타러너만 재학습

```bash
python main.py train-meta
```

### 15.5 전체 학습 파이프라인

```bash
python main.py train-full --sample-users 3000
```

### 15.6 모델 최적화

```bash
python main.py optimize-models
```

### 15.7 3D 애니 라이브러리 생성

```bash
python main.py visualize-3d --output artifacts/anime_tsne_3d.html
```

## 16. 3D 애니 라이브러리

3D 애니 라이브러리는 추천 모델의 핵심 서비스라기보다, 콘텐츠 feature가 실제로 비슷한 작품끼리 어느 정도 가까이 모이는지 보여주는 시각화 결과물이다.

생성 과정은 다음과 같다.

1. `meta_preprocessed.csv`를 불러온다.
2. feature를 `StandardScaler`로 표준화한다.
3. PCA로 차원을 1차 축소한다.
4. 3D t-SNE로 3차원 좌표를 만든다.
5. Plotly로 인터랙티브 HTML을 생성한다.

이를 통해 장르나 콘텐츠 특징이 비슷한 애니들이 3D 공간에서 어떻게 모이는지 확인할 수 있다.

## 17. 웹 서비스 구조

웹은 모델 결과를 보여주기 위한 인터페이스이며, 프로젝트의 핵심은 모델 파이프라인이다.

### 17.1 콘텐츠 기반 데모

- 사용자가 본 애니를 검색하고 선택한다.
- 선택한 애니들의 콘텐츠 유사도를 기반으로 비슷한 작품을 추천한다.
- 같은 시리즈를 제외하는 옵션을 제공한다.

### 17.2 회원제 하이브리드 데모

- 회원가입/로그인 기능을 제공한다.
- 사용자가 애니를 검색하고 1~10점으로 평가한다.
- 10개 이상 평가하면 SVD + CB + LightGBM 기반 개인 추천을 보여준다.
- 조회수는 SQLite counter로 저장한다.

## 18. 한계점과 개선 방향

### 18.1 한계점

- `anime_test.csv`의 컬럼 구조가 학습용 `anime.csv`와 달라 외부 테스트셋에서 CB feature를 완전히 재현하기 어려웠다.
- SVD와 LightGBM은 저장 파일이 커서 서버에서 바로 로드하면 메모리 부담이 크다.
- 신규 유저의 평점은 즉시 fold-in 방식으로 반영되지만, 원본 SVD 자체가 매번 재학습되는 것은 아니다.
- 콘텐츠 기반 추천은 메타데이터 품질에 영향을 많이 받는다.

### 18.2 개선 방향

- 신규 회원 평점이 충분히 쌓이면 주기적으로 SVD와 메타러너를 batch retraining한다.
- `anime_test.csv`에도 동일한 전처리 pipeline을 적용할 수 있도록 컬럼 스키마를 맞춘다.
- 제목 검색 외에 장르, 분위기, 러닝타임, 연령등급 기반 필터를 추가한다.
- 모델 설명성을 위해 SHAP으로 LightGBM이 CF와 CB 중 어느 쪽을 더 많이 참고했는지 분석한다.
- 추천 결과에 “왜 추천됐는지” 설명을 붙인다.

## 19. 최종 결론

이 프로젝트는 단순히 인기 애니를 추천하는 방식이 아니라, 유저의 평점 패턴과 애니의 콘텐츠 특징을 함께 고려하는 하이브리드 추천 시스템이다. SVD는 유저-아이템 평점 행렬에서 행동 패턴을 학습하고, 콘텐츠 기반 모델은 장르/제작사/스튜디오/인기도 등 애니 자체의 특징을 반영한다. 최종적으로 LightGBM 메타러너가 두 점수를 결합해 더 안정적인 개인화 추천을 만든다.

최종 모델은 `SVD + Content-Based Similarity + LightGBM Meta-Learner` 구조이며, 서버 배포를 위해 `svd_light.pkl`, `meta_model_core.pkl`, `item_sim_matrix.float32.npy`로 메모리 최적화까지 수행했다.
