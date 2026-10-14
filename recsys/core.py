r"""
저장된 모델 로드 + 평점 예측 predict(user_id, movie_id) / 추천 recommend(user_id, k) (API·CLI 공용)
파일 위치: D:\ysu_26_2\MLOPS\recsys\core.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  python recsys/core.py --input recsys/inputs/normal.json        # 정상 예측
  python recsys/core.py --input recsys/inputs/unknown_user.json  # 학습에 없는 유저(fallback)
  python recsys/core.py --input recsys/inputs/bad.json           # 잘못된 입력 -> JSON 에러
  python recsys/core.py --input recsys/inputs/recommend.json     # Top-N 추천(보조)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config
from recsys.models import BasePredictor  # noqa: F401  (joblib 역직렬화에 필요)

LIKE_THRESHOLD = config.RELEVANT_THRESHOLD


class InputError(ValueError):
    """사용자 입력 오류 (API 에서는 422/404)."""


class Recommender:
    def __init__(self, model_path: Path = config.MODEL_PATH):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {model_path} (python recsys/train.py 로 생성)")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.version = bundle["version"]
        self.data_hash = bundle["data_hash"]
        self._load_metadata()

    # ---------- 메타데이터 ----------
    def _load_metadata(self):
        movies = pd.read_csv(config.RAW_DIR / "movies.csv")
        links = pd.read_csv(config.RAW_DIR / "links.csv")
        movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
        if config.TOPRATED_CSV.exists():
            top = pd.read_csv(config.TOPRATED_CSV, usecols=["id", "overview"])
            top = top.drop_duplicates("id", keep="first")           # train_vote_average.py 와 같은 방식
            movies = movies.merge(top, left_on="tmdbId", right_on="id", how="left")
        else:
            movies["overview"] = np.nan
        self.movies = movies.set_index("movieId")
        self.titles = self.movies["title"].to_dict()
        self.genres = {m: ([] if g == "(no genres listed)" else g.split("|"))
                       for m, g in self.movies["genres"].items()}
        self.overviews = {m: o for m, o in self.movies["overview"].items() if isinstance(o, str) and o}

        train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
        test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
        self.train, self.test = train, test
        self.actual = {(int(u), int(m)): (float(r), "train") for u, m, r in train[["userId", "movieId", "rating"]].itertuples(index=False)}
        self.actual.update({(int(u), int(m)): (float(r), "test") for u, m, r in test[["userId", "movieId", "rating"]].itertuples(index=False)})
        counts = pd.concat([train, test]).groupby("userId").size()
        self._user_list = [{"user_id": int(u), "n_ratings": int(c)} for u, c in counts.items()]
        self._by_user_train = {int(u): g for u, g in train.groupby("userId")}

    def movie_info(self, movie_id: int) -> dict:
        return {
            "movieId": int(movie_id),
            "title": self.titles.get(movie_id, f"(unknown {movie_id})"),
            "genres": self.genres.get(movie_id, []),
            "tmdb_overview": self.overviews.get(movie_id),
        }

    # ---------- 기능 ----------
    def predict(self, user_id: int, movie_id: int) -> dict:
        if movie_id not in self.titles:
            raise InputError(f"movie_id {movie_id} 는 영화 목록에 없습니다")
        t0 = time.perf_counter()
        rating, fallback = self.model.predict(user_id, movie_id)
        out = {"user_id": user_id, **self.movie_info(movie_id),
               "predicted_rating": round(rating, 3), "likely_like": rating >= LIKE_THRESHOLD,
               "fallback": fallback, "fallback_reason": self._fallback_reason(user_id, movie_id) if fallback else None}
        act = self.actual.get((user_id, movie_id))
        out["actual_rating"] = act[0] if act else None
        out["actual_source"] = act[1] if act else None
        out["model_version"] = self.version
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 3)
        return out

    def _fallback_reason(self, user_id: int, movie_id: int) -> str:
        nu, ni = user_id not in self.model.user2idx, movie_id not in self.model.item2idx
        if nu and ni:
            return "학습에 없는 유저와 영화 → 전체 평균 사용"
        if nu:
            return "학습에 없는 유저 → 영화 평균 사용"
        return "학습에 없는 영화 → 유저 평균 사용"

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True) -> dict:
        t0 = time.perf_counter()
        recs = self.model.recommend(user_id, k=k, exclude_seen=exclude_seen)
        items = [{**self.movie_info(m), "predicted_rating": round(s, 3)} for m, s in recs]
        return {"user_id": user_id, "model_version": self.version,
                "fallback": user_id not in self.model.user2idx, "items": items,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    def users(self) -> list:
        return self._user_list

    def history(self, user_id: int, limit: int = 20) -> list:
        g = self._by_user_train.get(user_id)
        if g is None:
            return []
        g = g.sort_values(["rating", "timestamp"], ascending=[False, False]).head(limit)
        return [{**self.movie_info(int(m)), "rating": float(r)} for m, r in zip(g.movieId, g.rating)]

    def search_movies(self, q: str = "", limit: int = 20) -> list:
        df = self.movies
        if q:
            df = df[df["title"].str.contains(q, case=False, regex=False)]
        return [self.movie_info(int(m)) for m in df.index[:limit]]

    def samples(self, n: int = 30) -> list:
        """사람이 검증할 test 샘플: 시드 고정 무작위 n건 (학습에 있는 유저·영화만)."""
        t = self.test[self.test.userId.isin(self.model.user2idx) & self.test.movieId.isin(self.model.item2idx)]
        t = t.sample(n=min(n, len(t)), random_state=config.SEED)
        return [{"user_id": int(u), "movie_id": int(m), "title": self.titles.get(int(m)), "actual_rating": float(r)}
                for u, m, r in zip(t.userId, t.movieId, t.rating)]


_instance = None


def get_recommender() -> Recommender:
    global _instance
    if _instance is None:
        _instance = Recommender()
    return _instance


def validate_ids(data: dict, need_movie: bool) -> dict:
    """predict.py 와 같은 톤: 필수값·타입·범위 검사 후 InputError."""
    required = ["user_id"] + (["movie_id"] if need_movie else [])
    missing = [k for k in required if k not in data or data[k] in (None, "")]
    if missing:
        raise InputError(f"필수 입력 누락: {missing}")
    out = dict(data)
    for key in required:
        v = data[key]
        if isinstance(v, bool) or not isinstance(v, int):
            raise InputError(f"{key} 는 정수여야 합니다: {v!r}")
        if v < 1:
            raise InputError(f"{key} 는 1 이상이어야 합니다: {v}")
    if not need_movie:
        k = data.get("k", 10)
        if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 50:
            raise InputError(f"k 는 1~50 정수여야 합니다: {k!r}")
        out["k"] = k
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="평점 예측 / 추천 CLI")
    parser.add_argument("--input", required=True, type=Path, help="입력 JSON 경로")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    args = parser.parse_args()
    try:
        with open(args.input, encoding="utf-8") as f:
            data = json.load(f)
        rec = Recommender(args.model)
        if "movie_id" in data:
            d = validate_ids(data, need_movie=True)
            result = rec.predict(d["user_id"], d["movie_id"])
        else:
            d = validate_ids(data, need_movie=False)
            result = rec.recommend(d["user_id"], d["k"], d.get("exclude_seen", True))
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (InputError, FileNotFoundError, json.JSONDecodeError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
