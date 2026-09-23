"""
영화 평균 평점(vote_average) 예측 모델
파일 위치: D:\\ysu_26_2\\MLOPS\\train_vote_average.py

- 원본 CSV는 읽기만 하고 수정하지 않음
- 모델 3개 비교: 평균값 기준선 / Ridge(줄거리 TF-IDF 포함) / HistGradientBoosting(숫자·범주형)
- 가장 좋은 모델을 D:\\ysu_26_2\\MLOPS\\models\\ 에 저장
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

# ===== 설정 =====
BASE_DIR = Path(r"D:\ysu_26_2\MLOPS")
CSV_PATH = BASE_DIR / "top_rated_movies.csv"
MODEL_DIR = BASE_DIR / "models"

# popularity, vote_count는 평점과 같은 시점에 쌓이는 값이라
# 입력으로 쓸지 팀이 정해야 함. False로 바꾸면 두 열을 빼고 학습함
USE_VOTE_FEATURES = True

RANDOM_STATE = 42
TARGET = "vote_average"

# 명령줄로 CSV 경로를 넘기면 그걸 사용 (테스트용)
if len(sys.argv) > 1:
    CSV_PATH = Path(sys.argv[1])
    MODEL_DIR = CSV_PATH.parent / "models"


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"원본: {df.shape[0]}행 × {df.shape[1]}열")

    # id 중복: 원인 미확인 상태라 일단 첫 행만 남김 (원본 파일은 그대로)
    dup = df["id"].duplicated().sum()
    df = df.drop_duplicates(subset="id", keep="first").copy()
    print(f"id 중복 제거: {dup}행 → 남은 행 {len(df)}")

    # 결측 처리
    df["overview"] = df["overview"].fillna("")
    date = pd.to_datetime(df["release_date"], errors="coerce")
    print(f"release_date 변환 실패(결측 포함): {date.isna().sum()}행 → Ridge는 중앙값으로 채우고, HistGB는 결측 그대로 처리")
    df["release_year"] = date.dt.year
    df["release_month"] = date.dt.month

    # 치우친 분포라 로그 변환
    df["log_popularity"] = np.log1p(df["popularity"])
    df["log_vote_count"] = np.log1p(df["vote_count"])

    # 정답 결측 행은 제외 (현재 데이터는 0건)
    df = df.dropna(subset=[TARGET])
    return df


def build_models(num_cols, cat_cols, text_col):
    ridge_pre = ColumnTransformer([
        ("num", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10), cat_cols),
        ("text", TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=3), text_col),
    ])

    hgb_pre = ColumnTransformer([
        ("num", "passthrough", num_cols),  # HGB는 결측을 자체 처리함
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                               encoded_missing_value=-1), cat_cols),
    ])
    cat_mask = [False] * len(num_cols) + [True] * len(cat_cols)

    return {
        "Baseline(평균값)": DummyRegressor(strategy="mean"),
        "Ridge(+줄거리)": Pipeline([("pre", ridge_pre), ("model", Ridge(alpha=3.0))]),
        "HistGB": Pipeline([("pre", hgb_pre),
                            ("model", HistGradientBoostingRegressor(
                                categorical_features=cat_mask, max_iter=300,
                                learning_rate=0.05, random_state=RANDOM_STATE))]),
    }


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"{name:<18} MAE {mae:.4f} | RMSE {rmse:.4f} | R² {r2:.4f}")
    return mae


def main():
    df = load_data(CSV_PATH)

    num_cols = ["release_year", "release_month"]
    if USE_VOTE_FEATURES:
        num_cols += ["log_popularity", "log_vote_count"]
    cat_cols = []  # top_rated_movies.csv에는 original_language 열이 없음
    text_col = "overview"
    print(f"입력 열: {num_cols + cat_cols + [text_col]} (HistGB는 overview 제외)")
    print(f"정답 열: {TARGET}\n")

    X = df[num_cols + cat_cols + [text_col]]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)
    print(f"학습 {len(X_train)}행 / 평가 {len(X_test)}행\n")

    results = {}
    for name, model in build_models(num_cols, cat_cols, text_col).items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = (evaluate(name, y_test, pred), model, pred)

    best_name = min(results, key=lambda k: results[k][0])
    _, best_model, best_pred = results[best_name]
    print(f"\n최고 모델(MAE 기준): {best_name}")
    print(f"최고 모델 MAE: {results[best_name][0]:.4f} | 정확도(R²): {r2_score(y_test, best_pred):.4f}")

    sample = df.loc[X_test.index, ["title", TARGET]].copy()
    sample["예측"] = best_pred.round(3)
    sample["오차"] = (sample["예측"] - sample[TARGET]).round(3)
    print("\n평가 데이터 예측 예시 5건:")
    print(sample.head(5).to_string(index=False))

    pred_out = BASE_DIR / "vote_average_predictions.csv"
    sample.to_csv(pred_out, index=False, encoding="utf-8-sig")
    print(f"예측 결과 저장: {pred_out}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    out = MODEL_DIR / "vote_average_model.joblib"
    joblib.dump({"model": best_model, "name": best_name,
                 "features": num_cols + cat_cols + [text_col],
                 "use_vote_features": USE_VOTE_FEATURES}, out)
    print(f"\n모델 저장: {out}")


if __name__ == "__main__":
    main()
