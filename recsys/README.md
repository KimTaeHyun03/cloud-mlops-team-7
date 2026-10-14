# 영화 평점 예측 (recsys)

유저 ID + 영화 ID → 그 유저가 줄 평점(0.5~5.0) 예측. 같은 모델의 예측 평점 순으로 Top-N 추천도 제공(보조). 전부 로컬 실행, 배포 없음.
데이터: MovieLens ml-latest-small (GroupLens, 교육·연구용, 출처 표기 필수). 줄거리는 `top_rated_movies_utf8.csv`(TMDB)를 `links.csv`로 조인.

## 실행 방법 (PowerShell, 레포 루트 `D:\ysu_26_2\MLOPS`)

```powershell
pip install -r requirements.txt
python recsys/data_prep.py        # data/processed/train.csv, test.csv 생성
python recsys/train.py            # artifacts/model.joblib, metrics.json 생성
pytest recsys/tests               # 16개 테스트
uvicorn recsys.api:app --port 8000
# 브라우저: http://127.0.0.1:8000/    (API 문서: /docs)
```

CLI: `python recsys/core.py --input recsys/inputs/normal.json` (`unknown_user.json`, `bad.json`, `recommend.json`도 있음)

`artifacts/`, `data/processed/`는 git에 올리지 않는다 → 위 명령으로 재생성.

## Step 1. 데이터 준비 (`python recsys/data_prep.py` 실제 출력)

- ratings 100,836행 / 결측 0 / (userId, movieId) 중복 0
- 유저 610 / 평점 있는 영화 9,724 (movies.csv 9,742) / 희소도 98.3%
- 분할: 유저별 시간순 마지막 20% → train 80,672 / test 20,164
- test에만 있는 영화 1,485편: 학습 모델이 모르는 영화 → 예측은 "유저 평균"으로 대체하고 `fallback`으로 표시(평가에서 제외하지 않고 fallback 비율을 같이 기록)

## Step 2. 모델 비교 (`python recsys/train.py` 실제 출력, test 20,164건)

| 모델 | RMSE | MAE | P@10 | NDCG@10 | 설명 |
|---|---:|---:|---:|---:|---|
| global_mean | 1.0532 | 0.8246 | 0.0068 | 0.0089 | 전체 평균 |
| user_mean | 0.9665 | 0.7498 | 0.0068 | 0.0089 | 유저 평균 |
| item_mean | 0.9823 | 0.7600 | 0.0262 | 0.0365 | 영화 평균(축소) |
| svd | 0.9510 | 0.7345 | 0.0689 | 0.0919 | 유저평균 중심 SVD (32차원) |
| **svd_bias** | **0.9051** | **0.6927** | 0.0272 | 0.0314 | 개선판 (Step 6) |

- 같은 명령을 2번 실행해 동일한 지표가 나옴(시드 고정).
- 성공 기준: SVD 계열 RMSE < 최선 베이스라인(user_mean 0.9665) → **달성** (svd 0.9510, svd_bias 0.9051).
- test 중 8.39%는 학습에 없는 유저/영화라 평균으로 대체한 예측(`fallback`).

## Step 4. API (확인한 결과)

| 확인 | 결과 |
|---|---|
| `POST /predict {"user_id":1,"movie_id":318}` | 200, 응답 약 1ms 이내(서버 측 `latency_ms`) |
| `{"user_id":"abc",...}`, `user_id=0`, `movie_id` 누락, `movie_id=1.5` | 422 |
| 학습에 없는 유저 99999 | 200 + `fallback: true`(영화 평균 사용) |
| 영화 목록에 없는 `movie_id` | 404 |
| `POST /recommend` `k=100` / `k=0` / `k="x"` | 422 |
| `pytest recsys/tests` | 16 passed |

엔드포인트: `GET /health`, `POST /predict`, `POST /recommend`, `GET /movies?q=`, `GET /users`, `GET /users/{id}/history`, `GET /samples`, `GET /`.

## Step 5. 화면 — 사람이 확인할 시나리오

> **TODO(확인 필요):** 이 작업 중에는 브라우저 확장이 연결되지 않아 화면을 직접 열어보지 못했다. 정적 파일 서빙(200)과 화면이 호출하는 API, JS 문법 검사(`node --check`)만 확인했다. 아래를 사람이 눈으로 확인하고 결과를 적는다.

| 시나리오 | 기대 | 확인 결과 |
|---|---|---|
| 유저 1, 영화 318 → 예측하기 | 예상 평점 카드 표시 | ☐ |
| 유저 99999 | `fallback` 배지 + 사유 표시 | ☐ |
| 유저 `abc` / 영화 ID `abc` | 한글 오류 메시지, 화면 안 깨짐 | ☐ |
| 추천 k=0, k=100 | 한글 오류 메시지 | ☐ |
| "3. 검증" → 샘플 30건 전체 예측 | 예측·실제·오차 표 + 평균 오차 요약 | ☐ |
| 추천 카드 클릭 | 그 영화로 평점 예측 | ☐ |

## Step 6. 오차 분석과 개선

**분석 (svd, test):**
- 학습에서 평점 수가 적은 영화일수록 RMSE가 큼: 평점 0개 1.002 / 1~4개 0.977 / 5~19개 0.940 / 20~99개 0.942 / 100개 이상 0.924
- 잠재차원이 클수록 과적합: 8차원 0.9456 / 16 → 0.9475 / 32 → 0.9510 / 64 → 0.9570

**개선:** 유저 평균에 더해 **영화 편향**(평점 수가 적으면 0 쪽으로 축소)을 먼저 빼고 잔차에 SVD, 차원 8로 축소 (`svd_bias`).

| | RMSE | MAE |
|---|---:|---:|
| 개선 전 (svd, 32차원) | 0.9510 | 0.7345 |
| 개선 후 (svd_bias, 8차원) | 0.9051 | 0.6927 |

- 하이퍼파라미터(차원 4/8/16, 축소 2/5/10)는 RMSE 0.904~0.912 범위로 차이가 작아 기본값(8, 5)을 유지했다. 이 탐색은 test로 했으므로 엄밀한 일반화 추정은 아니다.
- **한계(트레이드오프):** Top-N 추천 보조 지표는 svd 0.0919 → svd_bias 0.0314(NDCG@10)로 **나빠졌다.** 영화 편향이 큰 영화(평점 10개 이상)가 상위에 올라오기 때문으로 추정되며, 아직 검증하지 않았다. 주 기능은 평점 예측이라 RMSE 기준으로 모델을 선정했다. 추천을 중시한다면 `/recommend`만 svd를 쓰는 방안이 있다 (미적용).
