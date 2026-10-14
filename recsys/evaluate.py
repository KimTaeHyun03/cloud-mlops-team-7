r"""
평점 예측 평가: RMSE, MAE (+ 보조: 추천 Precision/NDCG@K)
파일 위치: D:\ysu_26_2\MLOPS\recsys\evaluate.py

직접 실행하지 않고 train.py 가 import 한다.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config


def evaluate(model, test_df, k: int = config.TOP_K) -> dict:
    """test 전체에 대해 예측(클리핑 후) 오차 계산. 학습에 없는 유저/영화는 대체값으로 예측하고 비율을 기록."""
    pred, fallback = model.predict_many(test_df.userId.to_numpy(), test_df.movieId.to_numpy())
    err = pred - test_df.rating.to_numpy()
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "fallback_ratio": float(fallback.mean()),
        "n_test": int(len(test_df)),
    }
    out.update(_ranking_metrics(model, test_df, k))
    return out


def _ranking_metrics(model, test_df, k: int) -> dict:
    """보조 지표: 예측 평점 순 Top-k 가 test의 좋아한 영화(rating>=임계값)를 얼마나 맞히는가."""
    catalog = set(model.items.tolist())
    rel = test_df[(test_df.rating >= config.RELEVANT_THRESHOLD) & test_df.movieId.isin(catalog)]
    truth = rel.groupby("userId")["movieId"].apply(set)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    prec, ndcg = [], []
    for user_id, relevant in truth.items():
        recs = [m for m, _ in model.recommend(int(user_id), k=k, exclude_seen=True)]
        hits = np.array([m in relevant for m in recs], dtype=float)
        prec.append(hits.sum() / k)
        idcg = discounts[:min(len(relevant), k)].sum()
        ndcg.append((hits * discounts[:len(hits)]).sum() / idcg)
    return {f"precision@{k}": float(np.mean(prec)), f"ndcg@{k}": float(np.mean(ndcg))}
