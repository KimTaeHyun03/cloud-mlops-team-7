r"""
모델 학습 + 평가 + 비교표 출력 + 베스트 모델 저장
파일 위치: D:\ysu_26_2\MLOPS\recsys\train.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  python recsys/data_prep.py   # 먼저 1회
  python recsys/train.py

출력: artifacts/model.joblib, artifacts/metrics.json
"""
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config
from recsys.evaluate import evaluate
from recsys.models import SVD, GlobalMean, ItemMean, SVDBias, UserMean


def main() -> None:
    np.random.seed(config.SEED)
    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    k = config.TOP_K

    candidates = [
        (GlobalMean(), {}),
        (UserMean(), {}),
        (ItemMean(), {"damping": config.ITEM_DAMPING}),
        (SVD(), {"n_factors": config.SVD_FACTORS}),
        (SVDBias(), {"n_factors": config.SVD_BIAS_FACTORS, "damping": config.BIAS_DAMPING}),
    ]
    results = []
    for model, params in candidates:
        t0 = time.time()
        model.fit(train)
        fit_s = time.time() - t0
        m = evaluate(model, test, k=k)
        m["fit_seconds"] = fit_s
        results.append((model, params, m))

    print(f"\n{'model':<14}{'RMSE':>8}{'MAE':>8}{'P@10':>8}{'NDCG@10':>9}{'fallback':>10}{'fit(s)':>8}")
    for model, _, m in results:
        print(f"{model.name:<14}{m['rmse']:>8.4f}{m['mae']:>8.4f}{m[f'precision@{k}']:>8.4f}"
              f"{m[f'ndcg@{k}']:>9.4f}{m['fallback_ratio']:>10.4f}{m['fit_seconds']:>8.1f}")

    best_model, best_params, best_m = min(results, key=lambda r: r[2]["rmse"])
    baselines = [r for r in results if r[0].name not in ("svd", "svd_bias")]
    best_base = min(baselines, key=lambda r: r[2]["rmse"])
    print(f"\n베스트(RMSE 최소): {best_model.name} {best_m['rmse']:.4f} / 최선 베이스라인 {best_base[0].name} {best_base[2]['rmse']:.4f}")
    svd_m = next(r[2] for r in results if r[0] is best_model or r[0].name == "svd")
    if svd_m["rmse"] >= best_base[2]["rmse"]:
        print("경고: SVD 가 베이스라인보다 RMSE 가 낮지 않습니다. 원인을 README에 기록하세요.")

    now = datetime.now()
    data_hash = hashlib.sha256((config.PROCESSED_DIR / "train.csv").read_bytes()).hexdigest()[:12]
    version = f"{best_model.name}-{now:%Y%m%d}-{data_hash[:6]}"
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "version": version, "data_hash": data_hash}, config.MODEL_PATH)
    metrics = {
        "model_version": version,
        "model_name": best_model.name,
        "params": best_params,
        "metrics": best_m,
        "all_models": {m_.name: {"params": p, "metrics": mm} for m_, p, mm in results},
        "data_hash": data_hash,
        "trained_at": now.isoformat(timespec="seconds"),
        "seed": config.SEED,
        "relevant_threshold": config.RELEVANT_THRESHOLD,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"저장: {config.MODEL_PATH}, {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
