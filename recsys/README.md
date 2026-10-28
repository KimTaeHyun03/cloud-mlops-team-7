# 영화 평점 예측 (recsys)

유저 ID + 영화 ID → 그 유저가 줄 평점(0.5~5.0) 예측. 같은 모델의 예측 평점 순으로 Top-N 추천도 제공(보조). 전부 로컬 실행, 배포 없음.
데이터: MovieLens ml-latest-small (GroupLens, 교육·연구용, 출처 표기 필수). 줄거리는 `top_rated_movies_utf8.csv`(TMDB)를 `links.csv`로 조인.

## 실행 방법 (PowerShell, 레포 루트 `D:\ysu_26_2\MLOPS`)

```powershell
pip install -r requirements.txt
python recsys/data_prep.py        # data/processed/train.csv, test.csv 생성
python recsys/train.py            # artifacts/model.joblib, metrics.json 생성
pytest recsys/tests               # 26개 테스트
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
| `pytest recsys/tests` | 26 passed (내 평점 추천 포함) |

엔드포인트: `GET /health`, `POST /predict`, `POST /recommend`, `POST /recommend/from-ratings`, `GET /popular`, `GET /movies?q=`, `GET /users`, `GET /users/{id}/history`, `GET /samples`, `GET /`.

## Step 5. 화면 — 사람이 확인할 시나리오

화면(`/`)은 두 패널만 남겼다: **1. 내 평점으로 추천받기**, **2. 검증(예측 vs 실제 평점)**. 유저 ID로 예측·추천하는 화면 패널은 제거했고, API(`/predict`, `/recommend`)는 그대로 남아 있다(`/docs`에서 사용 가능).

> **TODO(확인 필요):** 이 작업 중에는 브라우저 확장이 연결되지 않아 화면을 직접 열어보지 못했다. 정적 파일 서빙(200), 화면이 호출하는 API, JS 문법 검사(`node --check`)만 확인했다. 아래를 사람이 눈으로 확인하고 결과를 적는다.

| 시나리오 | 기대 | 확인 결과 |
|---|---|---|
| "인기 영화에서 고르기" → 3개 이상 평점 선택 → 추천 받기 | 비슷한 유저 5명 카드 + 추천 10편 표시, 내가 준 영화는 제외 | ☐ |
| 영화 검색(예: Matrix)으로 평점 주기 | 검색 결과에서 평점 선택 가능 | ☐ |
| 평점 2개만 두고 추천 받기 | 한글 오류 메시지 (3개 이상 필요) | ☐ |
| 새로고침 | 내가 준 평점이 유지됨(브라우저 localStorage) | ☐ |
| "모두 지우기" | 평점 목록이 비워짐 | ☐ |
| "2. 검증" → 샘플 30건 전체 예측 | 예측·실제·오차 표 + 평균 오차 요약 | ☐ |

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

## 내 평점으로 추천 (fold-in)

화면 3번 패널. 유저가 영화에 평점을 주면 **재학습 없이** 학습된 `svd_bias` 모델에 새 유저를 끼워 넣는다.

1. 내가 준 평점 r 에서 영화 편향을 빼 평균 성향을 구함 (평점 수가 적으면 전체 평균 쪽으로 당김)
2. 남은 잔차를 학습된 영화 벡터 V 로 설명하는 내 취향 벡터 f 를 릿지 회귀로 계산: `f = (VᵀV + λI)⁻¹ Vᵀe`
3. 내가 안 본 영화(학습 평점 50개 이상) **전부**의 예상 평점 = 내 평균 + 영화 편향 + V·f → 높은 순 Top-10
4. 비슷한 유저 = 취향 벡터(f)와 기존 유저 벡터의 코사인 유사도 상위 5명

로그인은 없다. 내 평점은 브라우저(localStorage)에 저장되고, 요청할 때마다 서버로 보낸다(서버는 저장하지 않음).

**규제 강도 λ=0.06 선택 근거** (기존 유저의 train 평점 n개만 입력 → 그 유저의 test 평점 RMSE, 유저 300명, 작은 λ는 과적합):

| 입력 평점 수 | 취향 벡터 없이(평균+영화편향) | fold-in λ=0.06 |
|---|---:|---:|
| 5 | 0.9465 | 0.9455 |
| 10 | 0.9349 | 0.9322 |
| 20 | 0.9095 | 0.9039 |
| 전체 | 0.9034 | 0.8940 |

λ 후보(1e-4~0.5)를 test 로 골랐으므로 엄밀한 일반화 추정은 아니다. 평점이 5~10개일 때는 개인화 효과가 작다(평점을 더 줄수록 좋아짐).

**추천 순위 품질 (한계):** 같은 방식으로 기존 유저의 평점 n개만 입력하고 test 에서 좋아한(≥4.0) 영화를 맞히는지, 인기순(평점 수×평균, 같은 조건으로 평점 50개 이상·입력 영화 제외)과 비교 (300명):

| 입력 평점 수 | 내 평점 추천 P@10 / NDCG@10 | 인기순 P@10 / NDCG@10 |
|---|---:|---:|
| 10 | 0.0267 / 0.0393 | 0.0297 / 0.0379 |
| 20 | 0.0293 / 0.0381 | 0.0313 / 0.0413 |
| 50 | 0.0343 / 0.0448 | 0.0363 / 0.0482 |

- **인기순과 비슷하거나 약간 낮다.** 평점 예측용 모델(RMSE 최적화)이라 "무엇을 볼지" 순위 맞히기에는 강하지 않다. 이 한계는 Step 6 의 NDCG 하락과 같은 원인으로 추정(검증 안 함).
- 후보를 평점 50개 이상 영화로 제한한 것은 효과가 있었다 (제한 10개일 때 NDCG@10 0.0261 → 50개 0.0381, 입력 20개 기준; 이 값도 test 로 고름).
- **전부 예측 vs 랜덤 50편 후보:** 영화 8,239편 전부 예측해도 0.024ms(행렬곱 1회)라 비용 문제가 없고, 랜덤 50편은 오히려 느리고(0.214ms) 품질도 낮았다(NDCG@10 0.0286 vs 0.0381) → 전부 예측 채택.

## 사용 기록 (요청 로그) — 8주차
`POST /predict`, `/recommend`, `/recommend/from-ratings` 한 건당 JSON 한 줄을 stdout 과 `logs/requests.jsonl` 에 남긴다 (`logs/` 는 커밋 안 함).
응답 본문(성공 시)과 `X-Request-ID` 헤더(오류 포함)의 `request_id` 로 같은 요청의 로그를 찾는다.

| 필드 | 내용 |
|---|---|
| `request_id`, `timestamp`(UTC) | 요청 표식, 발생 시각 |
| `endpoint`, `status`, `status_code`, `error_type` | 처리 상태 (`validation_error`/`request_error`) |
| `total_ms` / `model_ms` | 전체 응답 시간 / 모델 계산 시간 (구별해서 기록) |
| `model_version`, `input`, `prediction` | 어떤 모델이 어떤 입력에 무엇을 답했나 |

- 이름·연락처·토큰은 기록하지 않는다. 내 평점 추천은 평점 개수(`n_ratings`)만 남기고 영화별 평점은 남기지 않는다.
- 실제 정답(`actual_rating`)을 확인해 연결하는 일은 아직 안 했다.
- 확인: `pytest recsys/tests` → 28 passed (요청 표식이 응답·헤더·로그에서 일치하는지 포함).

### CloudWatch Logs 전송 · S3 보관 (CloudShell)
로그 그룹은 `yeonsung_jkk`. 코드는 작성·테스트(가짜 클라이언트)했고, **실제 AWS 전송은 CloudShell 에서 실행해야 확인된다 (아직 미확인).**
로컬 자격증명은 IAM·S3 목록 조회가 막혀 있어 로컬에서는 AWS 를 호출하지 않는다.

1. 로컬: `python recsys/make_cloudshell_zip.py` → `dist/recsys-cloudshell.zip`
2. CloudShell(허용 리전, 정상 로그인): Actions → Upload file → ZIP 업로드 후
   `unzip recsys-cloudshell.zip -d recsys-lab && cd recsys-lab && python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt`
3. `cp recsys/config.example.json config.json` 후 `nano config.json` 에 `region`, `log_stream`, `s3_bucket`, `s3_prefix` 입력 (TODO(확인 필요): 허용 리전·S3 버킷·경로)
4. 서버: `RECSYS_LOG_CONFIG=config.json .venv/bin/python -m uvicorn recsys.api:app --host 127.0.0.1 --port 8080`
5. 새 탭에서 요청 전송 후 콘솔의 CloudWatch → `yeonsung_jkk` → 스트림에서 `request_id` 확인
6. S3 보관: `.venv/bin/python -m recsys.archive_logs --config config.json --start <UTC> --end <UTC>` (`--dry-run` 은 S3 에 쓰지 않음). 출력의 `reread.match: true` 와 건수·sha256 을 기록


{
  "region": "ap-northeast-2",
  "expected_account_id": "093356173512",
  "log_group": "yeonsung_jkk",
  "log_stream": "recsys-api-20261028",
  "send_interval_seconds": 5,
  "s3_bucket": "yeonsung-jkk-s3",
  "s3_prefix": "team7/recsys-logs>"
}
