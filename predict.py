"""
학습된 vote_average 모델로 새 입력 하나를 예측
파일 위치: D:\\ysu_26_2\\MLOPS\\predict.py

실행 명령 (PowerShell, 프로젝트 폴더에서):
  python predict.py --input inputs\\normal.json                                    # 정상 입력
  python predict.py --input inputs\\bad.json                                        # 잘못된 입력(숫자 자리에 문자열)
  python predict.py --input inputs\\normal.json --model models\\no-such-file.joblib  # 모델 파일 없음
  python predict.py --input inputs\\high-error-case.json                            # 오차 큰 실제 사례 재현

macOS/Linux(bash)에서는 경로 구분자를 /로 바꿔서 실행:
  python predict.py --input inputs/normal.json

필요 패키지: numpy, pandas, joblib, scikit-learn (미설치 시 pip install numpy pandas joblib scikit-learn)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

BASE_DIR = Path(r"D:\ysu_26_2\MLOPS")
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "vote_average_model.joblib"

REQUIRED_FIELDS = ["release_year", "release_month", "popularity", "vote_count", "overview"]


def load_input(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    missing = [k for k in REQUIRED_FIELDS if k not in data or data[k] in (None, "")]
    if missing:
        raise ValueError(f"필수 입력 누락: {missing}")

    for k in ["release_year", "release_month", "popularity", "vote_count"]:
        try:
            data[k] = float(data[k])
        except (TypeError, ValueError):
            raise ValueError(f"'{k}' 값이 숫자가 아닙니다: {data[k]!r}")

    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="입력 JSON 파일 경로")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), help="모델 joblib 경로")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(json.dumps({"error": f"모델 파일을 찾을 수 없습니다: {model_path}"}, ensure_ascii=False))
        sys.exit(1)

    try:
        data = load_input(Path(args.input))
    except ValueError as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False))
        sys.exit(1)

    saved = joblib.load(model_path)
    model = saved["model"]
    features = saved["features"]

    row = {
        "release_year": data["release_year"],
        "release_month": data["release_month"],
        "log_popularity": np.log1p(data["popularity"]),
        "log_vote_count": np.log1p(data["vote_count"]),
        "overview": data["overview"],
    }
    X = pd.DataFrame([row])[features]

    pred = model.predict(X)[0]
    print(json.dumps({
        "model_name": saved["name"],
        "input": data,
        "vote_average_pred": round(float(pred), 3),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
