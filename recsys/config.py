r"""
추천 시스템 공통 설정 (경로, 하이퍼파라미터, 랜덤시드)
파일 위치: D:\ysu_26_2\MLOPS\recsys\config.py

이 파일은 직접 실행하지 않고 다른 모듈이 import 한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "ml-latest-small"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
TOPRATED_CSV = ROOT / "top_rated_movies_utf8.csv"
STATIC_DIR = Path(__file__).resolve().parent / "static"

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

SEED = 42
RELEVANT_THRESHOLD = 4.0   # rating >= 이 값이면 "좋아함"
TEST_RATIO = 0.2           # 유저별 최근 20%를 test로
TOP_K = 10                 # 평가 K

# 모델 하이퍼파라미터
ITEM_DAMPING = 5           # 영화 평균을 전체 평균 쪽으로 당기는 가중치(가상 평점 수)
MIN_ITEM_RATINGS_FOR_REC = 10  # 추천(Top-N) 후보로 쓸 영화의 최소 학습 평점 수
SVD_FACTORS = 32
SVD_BIAS_FACTORS = 8       # 개선판(SVDBias) 잠재차원
BIAS_DAMPING = 5           # 개선판 영화 편향 축소 가중치
