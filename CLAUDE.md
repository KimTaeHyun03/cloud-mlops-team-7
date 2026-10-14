# CLAUDE.md — 영화 평점 예측 API (MLOps 7팀)

> **주제 변경 (2026-10-14):** 기존 "추천 Top-N" 중심에서 **"유저-영화 평점 예측"** 중심으로 변경. 주 기능은 `user_id + movie_id → 예상 평점`이고, 추천 Top-N은 같은 모델의 예측 평점 순 정렬로 얻는 **보조 기능**이다. 아래 본문에 "추천"으로 남은 표현은 이 변경에 맞춰 읽는다. 지표는 RMSE/MAE(주), 추천 보조 지표는 NDCG@10 등. 주제 변경에 맞춘 세부 규칙은 아래 "0. 평점 예측 주제 규칙"이 우선한다.

이 파일은 Claude Code가 이 레포에서 작업할 때 읽는 가이드다. 작업 전에 끝까지 읽고, 단계 순서대로 진행한다.

> ## ⚠️ 가장 중요: 이번 작업 범위
>
> **이번에 만들 것은 딱 두 가지다.**
>
> 1. **머신러닝 추천 서버** — 모델 학습 + 로컬에서 도는 FastAPI 추천 API
> 2. **화면** — 브라우저에서 유저 ID를 넣고 추천 결과를 보는 웹 화면
>
> **배포 관련 작업은 전부 하지 않는다.** Docker 이미지, ECR, EC2, AWS 리소스 생성, GitHub Actions, CloudWatch, Kubernetes/HPA, 부하 테스트 모두 **금지**. 요청받지 않았으면 Dockerfile이나 k8s 매니페스트 파일도 만들지 않는다.
>
> 모든 것은 로컬(`localhost`)에서 실행되면 완료다. 아래 Step 1~5만 진행하고, "나중 단계(하지 말 것)" 섹션은 참고용이다.

- 레포 루트: `D:\ysu_26_2\MLOPS` (GitHub: `cloud-mlops-team-7`, 작업 브랜치: `taehyun`)
- 환경: Windows / PowerShell / Python 3.10+
- 수업: 연성대 클라우드 MLOps 15주 프로젝트 (로드맵: `team-start-kit/COURSE_ROADMAP.md`)

---

## 0. 평점 예측 주제 규칙 (본문보다 우선)

**목표:** 유저 ID와 영화 ID를 넣으면 그 유저가 그 영화에 줄 평점(0.5~5.0)을 예측해 돌려준다. 화면은 예측 평점과 (test에 실제 평점이 있으면) 실제 평점을 나란히 보여줘 사람이 검증한다.

| 항목 | 내용 |
|---|---|
| 입력 | `user_id`(양의 정수), `movie_id`(양의 정수) |
| 출력 | `predicted_rating`(0.5~5.0으로 클리핑), 영화 제목·장르, `like_probability` 대신 `likely_like`(예측≥4.0), `fallback`, (있으면) `actual_rating`, `model_version`, `latency_ms` |
| 모델 | GlobalMean / UserMean / ItemMean(베이스라인) + 평균중심 SVD(행렬분해, numpy/scipy). 학습에 없는 유저·영화는 가능한 평균으로 대체하고 `fallback: true` |
| 지표 | RMSE, MAE (test 전체, 클리핑 후). **SVD의 RMSE가 베이스라인(UserMean/ItemMean 중 최선)보다 낮아야 성공.** 낮지 않으면 원인을 README에 기록 |
| 보조 | `POST /recommend`: 같은 모델의 예측 평점 순 Top-N(이미 본 영화 제외) |
| 검증(사람) | test.csv의 (user, movie, 실제 평점)을 화면에서 예측과 비교. 시나리오와 결과는 README에 기록 |
| 개선(Step 6) | 오차 큰 케이스(평점 수 적은 유저·영화 등)를 분석해 개선 포인트 1개 이상 적용, 전후 RMSE를 README에 표로 비교 |

엔드포인트(Step 4 표 대체): `GET /health`, `POST /predict`(주), `POST /recommend`(보조), `GET /movies`(검색·목록), `GET /users`, `GET /users/{id}/history`, `GET /samples`(test의 검증 샘플), `GET /`(화면). 배포는 여전히 금지.

---

## 1. 무엇을 만드나

**유저 ID를 넣으면 그 유저가 좋아할 영화 Top-N을 돌려주는 추천 API**, 그리고 이걸 컨테이너로 배포해 트래픽에 따라 확장·축소되는 것까지 보여주는 MLOps 프로젝트.

| 항목 | 내용 |
|---|---|
| 사용자 | 영화 서비스 이용자 (데모에서는 MovieLens의 userId 1~610) |
| 입력 | `user_id`, 추천 개수 `k` (선택: 이미 본 영화 제외 여부) |
| 출력 | 추천 영화 리스트 (`movieId`, 제목, 장르, 점수, 가능하면 TMDB 줄거리) |
| 성공 지표 | 오프라인: Precision@10, Recall@10, NDCG@10이 인기순 베이스라인보다 높음 / 온라인: p95 응답시간, 부하 시 오류율 |
| 이번 범위 | 재현 가능한 학습 → (선택) MLflow 실험 비교 → FastAPI 추천 서버 → 웹 화면. **전부 로컬 실행** |
| 나중 범위 (지금 금지) | Docker → 배포 → 자동배포·모니터링 → 오토스케일링 |

### 범위 밖 (하지 말 것)

- 딥러닝 추천 모델(NCF, Transformer 등). CPU로 10분 안에 학습되는 모델만 쓴다.
- 회원가입·로그인 같은 실제 서비스 기능
- 실시간 온라인 학습. 학습은 배치, 서빙은 미리 계산된 결과/행렬을 로드하는 방식
- **배포 일체** (Docker, AWS, CI/CD, K8s, 부하 테스트) — 이번 범위 아님

---

## 2. 데이터

모든 원본 데이터는 **읽기만 하고 절대 수정하지 않는다.** 가공 결과는 `artifacts/` 또는 `data/processed/`에 새 파일로 쓴다.

### 2-1. MovieLens ml-latest-small (메인)

경로: `D:\ysu_26_2\MLOPS\data\ml-latest-small\`
출처: https://grouplens.org/datasets/movielens/ (교육·연구용 가능, 출처 표기 필수, 상업적 사용 불가 — `README.txt` 참고)

| 파일 | 행 수 | 컬럼 |
|---|---:|---|
| `ratings.csv` | 100,836 | `userId, movieId, rating(0.5~5.0), timestamp` |
| `movies.csv` | 9,742 | `movieId, title, genres(`\|`구분)` |
| `links.csv` | 9,742 | `movieId, imdbId, tmdbId` (`tmdbId` 일부 결측) |
| `tags.csv` | 3,683 | `userId, movieId, tag, timestamp` |

- 유저 610명, 모든 유저가 최소 20개 이상 평점을 남김 (README 기준)

### 2-2. Top Rated Movies (보조, 기존 데이터)

경로: `D:\ysu_26_2\MLOPS\top_rated_movies_utf8.csv` (10,000행)
컬럼: `adult, id, original_language, original_title, overview, popularity, release_date, title, vote_average, vote_count`

- `id`는 **TMDB ID**다. `links.csv`의 `tmdbId`로 조인하면 약 5,498편이 매칭된다 (확인 완료).
- 용도: 추천 결과에 줄거리·원제·개봉일을 붙이는 표시용 메타데이터, (선택) 콘텐츠 기반 추천/콜드스타트용 특징
- 이 파일에 `id` 중복이 있다(기존 `train_vote_average.py`에서 첫 행만 남김). 조인 전에 같은 방식으로 중복 제거.

---

## 3. 기존 코드 (건드리지 말 것)

루트의 아래 파일은 4~5주차 과제(영화 평점 `vote_average` 회귀)이다. 새 추천 코드와 섞지 말고 그대로 둔다.

- `train_vote_average.py`, `predict.py`, `models/vote_average_model.joblib`, `inputs/*.json`, `vote_average_predictions.csv`
- `4주차/`, `5주차/`, `team-start-kit/` — 수업 자료

참고할 패턴: `predict.py`의 입력 검증 방식(필수값·타입·범위 검사 후 JSON 에러 반환)을 추천 API에도 같은 톤으로 적용한다.

---

## 4. 폴더 구조 (새로 만들 것)

```
D:\ysu_26_2\MLOPS\
├─ CLAUDE.md
├─ requirements.txt
├─ .gitignore                 # artifacts/, mlruns/, __pycache__/, .venv/ 제외
├─ data/ml-latest-small/      # 원본 (수정 금지)
└─ recsys/
   ├─ README.md               # 실행 방법 (다른 팀이 이것만 보고 재현 가능해야 함)
   ├─ config.py               # 경로·하이퍼파라미터·랜덤시드. 경로는 레포 루트 기준 상대경로로
   ├─ data_prep.py            # 로드·정제·train/test 분할 → data/processed/
   ├─ models.py               # Popularity / ItemKNN / SVD 추천기 (공통 fit/recommend 인터페이스)
   ├─ evaluate.py             # Precision@K, Recall@K, NDCG@K, Coverage
   ├─ train.py                # 학습+평가+MLflow 기록+베스트 모델 저장 → artifacts/
   ├─ core.py                 # 저장된 모델 로드 + recommend(user_id, k) 함수 (API·CLI 공용)
   ├─ api.py                  # FastAPI 앱 (API + 화면 서빙)
   ├─ static/                 # 화면: index.html, style.css, app.js
   ├─ tests/                  # pytest
   └─ inputs/                 # 정상·오류 요청 예시 JSON
```

Dockerfile, k8s/, loadtest/, .github/workflows/ 는 **만들지 않는다.**

**경로 규칙:** 기존 코드처럼 `D:\...` 절대경로를 하드코딩하지 말고 `Path(__file__).resolve().parents[1]` 기준으로 잡는다. 다른 팀원 PC에서도 그대로 돌아가야 한다. 단, 파일 상단 docstring에는 해당 파일의 절대경로(`D:\ysu_26_2\MLOPS\recsys\xxx.py`)를 적는다 (팀 규칙).

---

## 5. 단계별 작업

각 단계 끝에 **완료 조건**을 실제로 실행해서 확인하고, 결과(명령·출력)를 `recsys/README.md`에 남긴 뒤 커밋한다. 한 단계가 끝나기 전에 다음 단계로 넘어가지 않는다.

### Step 1. 환경·데이터 준비

- `requirements.txt`: `pandas, numpy, scipy, scikit-learn, joblib, fastapi, uvicorn[standard], pydantic, pytest, httpx` (버전 고정, `mlflow`는 Step 3 할 때만 추가)
- `scikit-surprise`는 Windows 빌드 문제가 잦으니 쓰지 않는다. numpy/scipy/sklearn만으로 구현.
- `data_prep.py`
  - 결측·중복 확인 결과 출력 (행 수, 유저 수, 영화 수, 희소도)
  - **분할: 유저별 시간순 분할** — 각 유저의 마지막 20%(또는 최근 N개) 평점을 test로. 랜덤 분할 금지(미래 정보 누수).
  - 암묵적 피드백 기준: `rating >= 4.0`을 "좋아함(relevant)"으로 정의 (config에서 바꿀 수 있게)
  - test에만 등장하는 영화는 평가에서 처리 방식 명시
- 완료 조건: `python recsys/data_prep.py` 실행 → 통계 출력, `data/processed/train.csv`, `test.csv` 생성

### Step 2. 베이스라인 + 모델 ★핵심

공통 인터페이스: `fit(train_df)`, `recommend(user_id, k, exclude_seen=True) -> list[(movieId, score)]`

1. **Popularity** (베이스라인): 평점 수 × 베이지안 평균으로 정렬. 모르는 유저에게도 이걸 반환 (콜드스타트 fallback).
2. **ItemKNN**: 유저-아이템 희소행렬(scipy.sparse) → 아이템 간 코사인 유사도 → 유저가 본 영화와 비슷한 영화 점수 합산
3. **SVD (행렬분해)**: 평균 중심화 후 `sklearn.decomposition.TruncatedSVD` 또는 `scipy.sparse.linalg.svds`. 잠재차원 수는 하이퍼파라미터.

- `evaluate.py`: test에서 relevant인 영화 기준으로 Precision@10, Recall@10, NDCG@10, 카탈로그 Coverage 계산
- 랜덤시드 고정. 같은 명령 두 번 실행 시 같은 지표가 나와야 함.
- 완료 조건: `python recsys/train.py` → 3개 모델 지표 비교표 출력, 베스트 모델이 `artifacts/model.joblib` + `artifacts/metrics.json`(지표·데이터 해시·학습 시각·파라미터)으로 저장. 베스트 모델이 Popularity보다 NDCG@10이 높아야 함. 낮으면 원인을 README에 기록.

### Step 3. (선택) MLflow 실험 기록 — 우선순위 낮음, Step 4·5 끝난 뒤 시간 남으면

- `train.py`에서 모델·하이퍼파라미터 조합마다 run 기록: params(모델명, k, 잠재차원, 이웃 수 등), metrics(P@10, R@10, NDCG@10, coverage, 학습시간), artifact(model.joblib)
- 기본은 로컬 `mlruns/` (`mlflow ui`로 확인). 원격 tracking 서버는 팀이 정하면 그때 붙인다.
- 완료 조건: `mlflow ui`에서 run 6개 이상 비교 가능, 모델 선정 근거를 README에 표로 정리

### Step 4. 추론 코어 + FastAPI 추천 서버 ★핵심

`core.py`: 시작 시 한 번만 모델·영화 메타데이터 로드 → `recommend(user_id, k)` 제공. CLI로도 실행 가능하게 (`python recsys/core.py --input recsys/inputs/normal.json`).

`api.py` 엔드포인트:

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/health` | 상태 + 로드된 모델 버전·해시 |
| POST | `/recommend` | body: `{"user_id": 1, "k": 10, "exclude_seen": true}` |
| GET | `/movies/{movie_id}/similar?k=10` | 비슷한 영화 (ItemKNN 또는 SVD 잠재벡터 유사도) |
| GET | `/users` | 데모용 유저 ID 목록 (화면 드롭다운용, 유저별 평점 수 포함) |
| GET | `/users/{user_id}/history?limit=20` | 유저가 높게 평가한 영화 (화면에서 "이 유저의 취향" 표시용) |
| GET | `/` | 화면(`static/index.html`) 서빙 |

응답 예시:

```json
{
  "user_id": 1,
  "model_version": "svd-20261014-abc123",
  "fallback": false,
  "items": [
    {"movieId": 318, "title": "Shawshank Redemption, The (1994)", "genres": ["Crime", "Drama"], "score": 4.71, "tmdb_overview": "..."}
  ],
  "latency_ms": 3.2
}
```

검증 규칙 (Pydantic):
- `user_id`: 양의 정수. 학습 데이터에 없는 유저면 에러가 아니라 Popularity 결과 + `"fallback": true`
- `k`: 1~50 정수, 기본 10
- 타입 오류·범위 초과 → 422

- 완료 조건: `uvicorn recsys.api:app --port 8000` 실행 후 정상 200, 오류 422, 미지 유저 fallback 3가지 확인. `pytest recsys/tests` 통과 (TestClient 사용). 단일 요청 응답 50ms 이내 목표 (로컬 기준).

### Step 5. 화면 (웹 UI) ★핵심

FastAPI가 `recsys/static/`을 서빙한다. 별도 프론트엔드 프레임워크(React 빌드 등) 없이 **HTML + CSS + 바닐라 JS** 로 만든다 (npm 설치 없이 바로 실행되도록).

화면 구성:
1. **입력 영역**: 유저 ID 입력(또는 `/users`로 채운 드롭다운) + 추천 개수(k) + "이미 본 영화 제외" 체크박스 + 추천 버튼. "랜덤 유저" 버튼 있으면 좋음
2. **이 유저의 취향**: `/users/{id}/history`로 유저가 높게 평가한 영화 몇 편 (제목·장르·평점)
3. **추천 결과**: 카드 목록 — 순위, 제목, 장르 태그, 점수, TMDB 줄거리(있으면, 길면 접기)
4. **카드 클릭 → 비슷한 영화**: `/movies/{id}/similar` 결과를 패널/모달로 표시
5. **상태 표시**: 로딩 중, 에러 메시지(422 내용을 사람이 읽을 수 있게), 학습에 없는 유저라 인기순으로 대체했으면 "fallback" 배지, 응답시간(ms), 모델 버전

요구사항:
- `fetch`는 같은 출처(`/recommend` 등 상대경로)로 호출 → CORS 설정 불필요
- 외부 CDN·외부 API 호출 없이 로컬만으로 동작
- 영화 포스터 이미지는 넣지 않는다 (TMDB API 키 필요 → 범위 밖)
- 노트북 화면 기준으로 깔끔하게, 모바일 폭에서도 깨지지 않게

완료 조건: `uvicorn recsys.api:app --port 8000` 실행 → 브라우저에서 `http://127.0.0.1:8000/` 접속 → 아래 시나리오를 직접 확인하고 결과를 README에 기록
- 정상 유저(예: 1번) 추천 표시
- 학습에 없는 유저(예: 99999) → fallback 배지와 함께 인기순 표시
- 잘못된 입력(문자, k=0, k=100) → 에러 메시지 표시, 화면 안 깨짐
- 카드 클릭 → 비슷한 영화 표시

**Step 4·5까지 되면 이번 작업은 끝이다.** 거기서 멈추고 사람에게 결과를 보고한다.

---

## 나중 단계 (지금은 하지 말 것 — 참고용)

아래는 수업 후반 주차에 할 일이다. **명시적으로 요청받기 전에는 시작하지 않고, 관련 파일도 만들지 않는다.**

- Docker 이미지 빌드, ECR 푸시
- EC2 배포
- GitHub Actions 자동 배포
- CloudWatch / Prometheus 모니터링
- Kubernetes Deployment·HPA 오토스케일링, Locust 부하 테스트

## 6. 작업 규칙

- **배포·클라우드 작업 금지** (Docker, AWS, CI/CD, K8s). AWS CLI·docker 명령을 실행하지 않는다.
- 원본 CSV 수정 금지. 기존 4~5주차 파일 수정 금지.
- 커밋은 단계별로 작게, 메시지는 한국어로 무엇을 했는지 한 줄. 브랜치 `taehyun`에서 작업, `master` 직접 푸시 금지.
- `artifacts/`, `mlruns/`, `.venv/`는 커밋하지 않는다 (`.gitignore`). 대신 재생성 명령을 README에 적는다.
- 확실하지 않은 것(AWS 계정 정보, 팀 결정 사항, 수업 제출 형식)은 추측해서 채우지 말고 `TODO(확인 필요)`로 남기고 사람에게 물어본다.
- 새 파일 상단 docstring에 파일 절대경로와 실행 명령(PowerShell 기준)을 적는다 — 기존 `predict.py` 형식 참고.
- 지표를 보고할 때는 실제 실행 출력만 쓴다. 실행 안 한 숫자를 쓰지 않는다.

## 7. 지금 상태 (2026-10-14)

- [x] MovieLens 다운로드·압축 해제 (`data/ml-latest-small/`, 공식 MD5 일치 확인)
- [x] Top Rated Movies ↔ MovieLens TMDB ID 조인 가능 확인 (5,498편)
- [ ] `data/`는 아직 git 미커밋 (약 3MB) — 커밋할지 팀이 결정
- [ ] Step 1부터 시작 → 목표는 Step 4(추천 서버) + Step 5(화면)까지. 배포는 하지 않음
