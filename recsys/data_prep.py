r"""
MovieLens 로드·정제·유저별 시간순 train/test 분할
파일 위치: D:\ysu_26_2\MLOPS\recsys\data_prep.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  python recsys/data_prep.py

출력: data/processed/train.csv, data/processed/test.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config


def load_ratings() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DIR / "ratings.csv")


def load_movies() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DIR / "movies.csv")


def split_by_user_time(ratings: pd.DataFrame, test_ratio: float = config.TEST_RATIO):
    """각 유저의 시간순 마지막 test_ratio 비율을 test로 (미래 정보 누수 방지)."""
    df = ratings.sort_values(["userId", "timestamp", "movieId"]).reset_index(drop=True)
    pos = df.groupby("userId").cumcount()
    n = df.groupby("userId")["movieId"].transform("size")
    n_test = (n * test_ratio).round().clip(lower=1).astype(int)
    is_test = pos >= (n - n_test)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


def main() -> None:
    ratings = load_ratings()
    movies = load_movies()

    n_users, n_items = ratings.userId.nunique(), ratings.movieId.nunique()
    sparsity = 1 - len(ratings) / (n_users * n_items)
    print("== 데이터 점검 ==")
    print(f"ratings 행 수: {len(ratings):,} / 결측: {int(ratings.isna().sum().sum())} / "
          f"(userId,movieId) 중복: {int(ratings.duplicated(['userId', 'movieId']).sum())}")
    print(f"유저 수: {n_users} / 평점 있는 영화 수: {n_items} / movies.csv 영화 수: {len(movies)}")
    print(f"희소도: {sparsity:.4%}")

    train, test = split_by_user_time(ratings)
    n_test_only = len(set(test.movieId) - set(train.movieId))
    print("\n== 분할 (유저별 시간순, 마지막 20%) ==")
    print(f"train: {len(train):,} / test: {len(test):,}")
    print(f"test에만 등장하는 영화: {n_test_only}편 "
          f"(처리: 학습에 없어 모델이 추천할 수 없으므로 평가용 정답(relevant)에서 제외)")
    rel = test[test.rating >= config.RELEVANT_THRESHOLD]
    print(f"test relevant(rating>={config.RELEVANT_THRESHOLD}) 평점 수: {len(rel):,} / 유저 수: {rel.userId.nunique()}")

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(config.PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(config.PROCESSED_DIR / "test.csv", index=False)
    print(f"\n저장: {config.PROCESSED_DIR}")


if __name__ == "__main__":
    main()
