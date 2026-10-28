r"""
CloudShell 에 올릴 ZIP 만들기 (코드 + 모델 + 실행에 필요한 데이터만)
파일 위치: D:\ysu_26_2\MLOPS\recsys\make_cloudshell_zip.py

실행 명령 (PowerShell, 프로젝트 루트에서):
  python recsys/make_cloudshell_zip.py      # → dist/recsys-cloudshell.zip
원본 데이터는 읽기만 하고 수정하지 않는다. 학습 전 artifacts/·data/processed/ 가 먼저 있어야 한다(README 참고).
"""
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config

FILES = [
    config.ROOT / "requirements.txt",
    config.TOPRATED_CSV,
    config.RAW_DIR / "movies.csv",
    config.RAW_DIR / "links.csv",
    config.MODEL_PATH,
    config.METRICS_PATH,
    config.PROCESSED_DIR / "train.csv",
    config.PROCESSED_DIR / "test.csv",
]


def main() -> None:
    missing = [p for p in FILES if not p.exists()]
    if missing:
        raise SystemExit("없는 파일 (python recsys/data_prep.py, python recsys/train.py 먼저): " + ", ".join(str(m) for m in missing))
    out = config.ROOT / "dist" / "recsys-cloudshell.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((config.ROOT / "recsys").rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and "tests" not in p.parts and p.suffix != ".pyc":
                z.write(p, p.relative_to(config.ROOT).as_posix())
        for p in FILES:
            z.write(p, p.relative_to(config.ROOT).as_posix())
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
