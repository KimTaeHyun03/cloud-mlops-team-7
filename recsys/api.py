r"""
FastAPI 평점 예측 서버 (API + 화면 서빙)
파일 위치: D:\ysu_26_2\MLOPS\recsys\api.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  uvicorn recsys.api:app --port 8000
  # 브라우저: http://127.0.0.1:8000/   API 문서: http://127.0.0.1:8000/docs
"""
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config
from recsys.core import InputError, Recommender

app = FastAPI(title="영화 평점 예측 API", version="1.0")
rec = Recommender()   # 시작 시 한 번만 모델·메타데이터 로드


class PredictRequest(BaseModel):
    user_id: int = Field(ge=1, description="유저 ID (양의 정수)")
    movie_id: int = Field(ge=1, description="영화 ID (양의 정수)")


class RecommendRequest(BaseModel):
    user_id: int = Field(ge=1)
    k: int = Field(default=10, ge=1, le=50)
    exclude_seen: bool = True


@app.get("/health")
def health():
    return {"status": "ok", "model_version": rec.version, "data_hash": rec.data_hash,
            "model_name": rec.model.name}


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return rec.predict(req.user_id, req.movie_id)
    except InputError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest):
    return rec.recommend(req.user_id, req.k, req.exclude_seen)


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
