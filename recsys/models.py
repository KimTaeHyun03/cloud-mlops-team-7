r"""
평점 예측 모델: GlobalMean / UserMean / ItemMean(베이스라인) + SVD(행렬분해)
파일 위치: D:\ysu_26_2\MLOPS\recsys\models.py

공통 인터페이스: fit(train_df), predict(user_id, movie_id) -> (rating, fallback),
                predict_many(user_ids, movie_ids) -> (ratings, fallback_mask),
                recommend(user_id, k, exclude_seen) -> list[(movieId, predicted_rating)]
직접 실행하지 않고 train.py / core.py 가 import 한다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config

MIN_RATING, MAX_RATING = 0.5, 5.0


class BasePredictor:
    name = "base"

    def fit(self, train_df: pd.DataFrame):
        self.users = np.sort(train_df.userId.unique())
        self.items = np.sort(train_df.movieId.unique())
        self.user2idx = {int(u): i for i, u in enumerate(self.users)}
        self.item2idx = {int(m): i for i, m in enumerate(self.items)}
        ui = train_df.userId.map(self.user2idx).to_numpy()
        ii = train_df.movieId.map(self.item2idx).to_numpy()
        r = train_df.rating.to_numpy(dtype=np.float64)
        self.R = sp.csr_matrix((r, (ui, ii)), shape=(len(self.users), len(self.items)))
        self.global_mean = float(r.mean())
        n_u = np.bincount(ui, minlength=len(self.users)).astype(np.float64)
        n_i = np.bincount(ii, minlength=len(self.items)).astype(np.float64)
        self.user_mean = np.bincount(ui, weights=r, minlength=len(self.users)) / np.maximum(n_u, 1)
        m = config.ITEM_DAMPING
        self.item_mean = (np.bincount(ii, weights=r, minlength=len(self.items)) + m * self.global_mean) / (n_i + m)
        self.n_user_ratings, self.n_item_ratings = n_u, n_i
        self._fit(ui, ii, r)
        return self

    def _fit(self, ui, ii, r):
        pass

    def _raw(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        """u, i: 인덱스 배열(-1이면 학습에 없음). 하위 클래스가 구현."""
        raise NotImplementedError

    def _fallback_value(self, u: np.ndarray, i: np.ndarray) -> np.ndarray:
        """학습에 없는 유저/영화일 때의 대체 예측."""
        out = np.full(len(u), self.global_mean)
        known_u, known_i = u >= 0, i >= 0
        out[known_i & ~known_u] = self.item_mean[i[known_i & ~known_u]]
        out[known_u & ~known_i] = self.user_mean[u[known_u & ~known_i]]
        return out

    def predict_many(self, user_ids, movie_ids):
        u = np.array([self.user2idx.get(int(x), -1) for x in user_ids])
        i = np.array([self.item2idx.get(int(x), -1) for x in movie_ids])
        known = (u >= 0) & (i >= 0)
        pred = self._fallback_value(u, i)
        if known.any():
            pred[known] = self._raw(u[known], i[known])
        return np.clip(pred, MIN_RATING, MAX_RATING), ~known

    def predict(self, user_id: int, movie_id: int):
        p, fb = self.predict_many([user_id], [movie_id])
        return float(p[0]), bool(fb[0])

    def _all_items_for_user(self, u: int) -> np.ndarray:
        n = len(self.items)
        return self._raw(np.full(n, u), np.arange(n))

    def recommend(self, user_id: int, k: int = 10, exclude_seen: bool = True):
        """예측 평점 순 Top-k -> list[(movieId, predicted_rating)]. 학습에 없는 유저는 영화 평균 순(fallback)."""
        u = self.user2idx.get(int(user_id))
        if u is None:
            scores = self.item_mean.copy()
            scores[self.n_item_ratings < config.MIN_ITEM_RATINGS_FOR_REC] = -np.inf
        else:
            scores = np.clip(self._all_items_for_user(u), MIN_RATING, MAX_RATING)
            scores[self.n_item_ratings < config.MIN_ITEM_RATINGS_FOR_REC] = -np.inf
            if exclude_seen:
                scores[self.R[u].indices] = -np.inf
        k = min(k, int(np.isfinite(scores).sum()))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(int(self.items[j]), float(scores[j])) for j in top]


class GlobalMean(BasePredictor):
    name = "global_mean"

    def _raw(self, u, i):
        return np.full(len(u), self.global_mean)


class UserMean(BasePredictor):
    name = "user_mean"

    def _raw(self, u, i):
        return self.user_mean[u]


class ItemMean(BasePredictor):
    name = "item_mean"

    def _raw(self, u, i):
        return self.item_mean[i]


class SVD(BasePredictor):
    """유저 평균 중심화 후 관측값만으로 truncated SVD. 예측 = 유저평균 + U·S·Vt."""
    name = "svd"

    def __init__(self, n_factors: int = config.SVD_FACTORS):
        self.n_factors = n_factors

    def _fit(self, ui, ii, r):
        centered = sp.csr_matrix((r - self.user_mean[ui], (ui, ii)), shape=self.R.shape)
        v0 = np.random.RandomState(config.SEED).uniform(-1, 1, min(centered.shape))
        U, s, Vt = svds(centered, k=self.n_factors, v0=v0)
        order = np.argsort(-s)
        U, s, Vt = U[:, order], s[order], Vt[order]
        self.user_f = U * s
        self.item_f = Vt.T

    def _raw(self, u, i):
        return self.user_mean[u] + np.einsum("ij,ij->i", self.user_f[u], self.item_f[i])


MODEL_CLASSES = {c.name: c for c in (GlobalMean, UserMean, ItemMean, SVD)}
