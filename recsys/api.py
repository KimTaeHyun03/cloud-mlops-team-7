r"""
FastAPI 평점 예측 서버 (API + 화면 서빙)
파일 위치: D:\ysu_26_2\MLOPS\recsys\api.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  uvicorn recsys.api:app --port 8000
  # 브라우저: http://127.0.0.1:8000/   API 문서: http://127.0.0.1:8000/docs
"""
import sys
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config
from recsys.core import InputError, Recommender
from recsys.request_log import log_request, setup as setup_request_log

app = FastAPI(title="영화 평점 예측 API", version="1.0")
rec = Recommender()   # 시작 시 한 번만 모델·메타데이터 로드
req_logger = setup_request_log()
LOGGED_ENDPOINTS = {"/predict", "/recommend", "/recommend/from-ratings"}   # 예측을 만드는 POST 요청만 기록


@app.middleware("http")
async def request_log_middleware(request: Request, call_next):
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    request.state.log_fields = {}
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id   # 오류 응답도 표식으로 로그를 찾을 수 있게 헤더에 남긴다
    if request.method == "POST" and request.url.path in LOGGED_ENDPOINTS:
        ok = response.status_code < 400
        log_request(req_logger, request_id=request_id, endpoint=request.url.path,
                    status="ok" if ok else "error", status_code=response.status_code,
                    error_type=None if ok else ("validation_error" if response.status_code == 422 else "request_error"),
                    total_ms=round((time.perf_counter() - t0) * 1000, 3),
                    **request.state.log_fields)
    return response


class PredictRequest(BaseModel):
    user_id: int = Field(ge=1, description="유저 ID (양의 정수)")
    movie_id: int = Field(ge=1, description="영화 ID (양의 정수)")


class RecommendRequest(BaseModel):
    user_id: int = Field(ge=1)
    k: int = Field(default=10, ge=1, le=50)
    exclude_seen: bool = True


class GuestRating(BaseModel):
    movie_id: int = Field(ge=1)
    rating: float = Field(ge=0.5, le=5.0, multiple_of=0.5)


class GuestRecommendRequest(BaseModel):
    ratings: list[GuestRating] = Field(min_length=3, max_length=200)
    k: int = Field(default=10, ge=1, le=50)


@app.post("/recommend/from-ratings")
def recommend_from_ratings(req: GuestRecommendRequest, request: Request):
    request.state.log_fields = {"input": {"n_ratings": len(req.ratings), "k": req.k}}
    ids = [r.movie_id for r in req.ratings]
    if len(set(ids)) != len(ids):
        raise HTTPException(status_code=422, detail="같은 영화에 평점이 중복되었습니다")
    try:
        out = rec.guest_recommend({r.movie_id: r.rating for r in req.ratings}, req.k)
        request.state.log_fields.update(
            model_version=out["model_version"], model_ms=out["latency_ms"],
            prediction={"n_ratings_used": out["n_ratings_used"], "n_items": len(out["items"])})
        return {**out, "request_id": request.state.request_id}
    except InputError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/popular")
def popular(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0, le=5000)):
    return rec.popular(limit, offset)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": rec.version, "data_hash": rec.data_hash,
            "model_name": rec.model.name}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    request.state.log_fields = {"input": {"user_id": req.user_id, "movie_id": req.movie_id}}
    try:
        out = rec.predict(req.user_id, req.movie_id)
        request.state.log_fields.update(
            model_version=out["model_version"], model_ms=out["latency_ms"],
            prediction={"predicted_rating": out["predicted_rating"], "fallback": out["fallback"]})
        return {**out, "request_id": request.state.request_id}
    except InputError as e:
        request.state.log_fields["error_detail"] = str(e)
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest, request: Request):
    out = rec.recommend(req.user_id, req.k, req.exclude_seen)
    request.state.log_fields = {
        "input": {"user_id": req.user_id, "k": req.k, "exclude_seen": req.exclude_seen},
        "model_version": out["model_version"], "model_ms": out["latency_ms"],
        "prediction": {"fallback": out.get("fallback"), "n_items": len(out["items"])}}
    return {**out, "request_id": request.state.request_id}


@app.get("/movies")
def movies(q: str = Query("", max_length=100), limit: int = Query(20, ge=1, le=100)):
    return rec.search_movies(q, limit)


@app.get("/users")
def users():
    return rec.users()


@app.get("/users/{user_id}/history")
def history(user_id: int, limit: int = Query(20, ge=1, le=100)):
    if user_id < 1:
        raise HTTPException(status_code=422, detail="user_id 는 1 이상이어야 합니다")
    return rec.history(user_id, limit)


@app.get("/samples")
def samples(n: int = Query(30, ge=1, le=200)):
    return rec.samples(n)


@app.get("/")
def index():
    return FileResponse(config.STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
