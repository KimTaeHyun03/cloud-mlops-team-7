r"""
피드백 로그 -> 학습 데이터 병합 -> 재학습 -> 고정 test 로 현재 모델과 비교 -> 통과하면 모델 교체
파일 위치: D:\ysu_26_2\MLOPS\recsys\retrain.py

실행 명령 (PowerShell, 프로젝트 루트에서. 먼저 python recsys/data_prep.py, python recsys/train.py 가 돌아 있어야 한다):
  python recsys/retrain.py                                   # logs/requests.jsonl 사용
  python recsys/retrain.py --log-file a.jsonl --log-file b.jsonl   # S3 에서 받은 data.jsonl 등 (여러 번 가능)
  python recsys/retrain.py --config config.json --start 2026-11-01T00:00:00Z --end 2026-11-11T00:00:00Z
                                                             # CloudWatch 에서 직접 조회 (CloudShell, 시각은 UTC)
  python recsys/retrain.py --force      # 새 피드백이 적어도 실행
  python recsys/retrain.py --dry-run    # 저장·교체 없이 결과만 확인

흐름
  1. 로그 읽기      파일 또는 CloudWatch. 한 줄 = JSON 한 개 (request_log.py 형식)
  2. 피드백 추출    event=feedback 에서 (user_id, movie_id, actual_rating). user_id 가 없는 옛 로그는 request_id 로
                    같은 로그의 요청 이벤트에서 찾는다. 내 평점 추천(guest)은 학습 유저가 아니라서 제외.
  3. 걸러내기      학습에 없는 유저·영화, 이미 원본 데이터에 있는 (유저, 영화) 쌍은 제외 (test 누수 방지)
  4. 누적 저장      data/feedback/feedback_ratings.csv 에 합치고 (유저, 영화)당 가장 늦은 평점만 남김
  5. 트리거        현재 모델이 아직 학습하지 않은 피드백이 RETRAIN_MIN_NEW_FEEDBACK 이상일 때만 (--force 로 무시)
  6. 재학습        현재 모델과 같은 종류·하이퍼파라미터로 train + 피드백 전체를 다시 학습
  7. 게이트        같은 test.csv(피드백은 절대 안 섞임)로 현재/후보 RMSE 비교.
                   후보가 RETRAIN_MAX_RMSE_REGRESSION 넘게 나쁘면 교체하지 않는다
  8. 교체          후보는 artifacts/candidates/<버전>/ 에 항상 남기고, 통과하면 이전 모델을 artifacts/previous/<버전>/ 에
                   백업한 뒤 artifacts/model.joblib, metrics.json 을 교체. 기록은 artifacts/retrain_history.jsonl
종료 코드: 0 = 교체함 또는 건너뜀 또는 dry-run, 2 = 후보가 게이트를 통과하지 못함 (CI 에서 감지용)

주의: 서버는 시작할 때 모델을 한 번 읽는다. 교체한 모델을 서비스에 쓰려면 서버(컨테이너)를 다시 시작해야 한다.
"""
import argparse
import hashlib
import inspect
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys import config
from recsys.evaluate import evaluate
from recsys.models import MODEL_CLASSES

COLUMNS = ["userId", "movieId", "rating", "timestamp"]
PREDICT_ENDPOINTS = {"/predict", "/recommend"}   # input 에 user_id 가 남는 요청 (from-ratings 는 유저가 없다)


# ---------- 1. 로그 읽기 ----------
def read_log_files(paths: list[Path]) -> list[str]:
    lines: list[str] = []
    for p in paths:
        lines += Path(p).read_text(encoding="utf-8").splitlines()
    return lines


def lines_from_cloudwatch(cfg: dict, start: datetime, end: datetime, logs=None) -> list[str]:
    """CloudWatch 의 [start, end) 기록 메시지(= JSON 한 줄)를 시간순으로 가져온다. archive_logs 와 같은 조회를 쓴다."""
    from recsys.archive_logs import fetch_messages
    if logs is None:
        from recsys.cloudwatch_sink import check_account, make_client
        check_account(cfg)
        logs = make_client(cfg, "logs")
    return fetch_messages(logs, cfg["log_group"], cfg["log_stream"], start, end)


def parse_events(lines: list[str]) -> tuple[list[dict], int]:
    events, bad = [], 0
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(ev, dict):
            events.append(ev)
        else:
            bad += 1
    return events, bad


# ---------- 2. 피드백 추출 ----------
def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _to_epoch(ts) -> int | None:
    try:
        return int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _valid_rating(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and 0.5 <= x <= 5.0 and (x * 2) == int(x * 2))


def empty_ratings() -> pd.DataFrame:
    return pd.DataFrame({"userId": pd.Series(dtype="int64"), "movieId": pd.Series(dtype="int64"),
                         "rating": pd.Series(dtype="float64"), "timestamp": pd.Series(dtype="int64")})


def extract_feedback(events: list[dict]) -> tuple[pd.DataFrame, Counter]:
    """feedback 이벤트 -> (userId, movieId, rating, timestamp). (유저, 영화)가 겹치면 가장 늦은 것만."""
    requests = {}
    for ev in events:
        if ev.get("event") == "request" and ev.get("status") == "ok" and ev.get("endpoint") in PREDICT_ENDPOINTS:
            uid = (ev.get("input") or {}).get("user_id")
            if _is_int(uid) and uid >= 1 and ev.get("request_id"):
                requests[ev["request_id"]] = uid

    stats: Counter = Counter()
    rows = []
    for ev in events:
        if ev.get("event") != "feedback":
            continue
        stats["feedback_events"] += 1
        uid = ev["user_id"] if _is_int(ev.get("user_id")) else requests.get(ev.get("request_id"))
        if uid is None or uid < 1:
            stats["skip_no_user"] += 1   # 내 평점 추천(guest)이거나, user_id 가 없는 옛 로그인데 요청 기록도 못 찾음
            continue
        movie, rating, ts = ev.get("movie_id"), ev.get("actual_rating"), _to_epoch(ev.get("timestamp"))
        if not _is_int(movie) or movie < 1 or not _valid_rating(rating) or ts is None:
            stats["skip_invalid"] += 1
            continue
        rows.append((uid, movie, float(rating), ts))

    if not rows:
        return empty_ratings(), stats
    df = pd.DataFrame(rows, columns=COLUMNS).sort_values("timestamp", kind="stable")
    deduped = df.drop_duplicates(["userId", "movieId"], keep="last").reset_index(drop=True)
    stats["skip_duplicate"] = len(df) - len(deduped)
    return deduped, stats


# ---------- 3. 걸러내기 ----------
def filter_feedback(fb: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                    movie_ids: set) -> tuple[pd.DataFrame, Counter]:
    """학습에 쓸 수 있는 피드백만 남긴다.
    - 원본에 없는 유저: 새 유저를 만들지 않는다 (평점 1~2개짜리 가짜 유저가 행렬을 흔든다)
    - 영화 목록에 없는 영화
    - 원본(train/test)에 이미 있는 (유저, 영화): 새 정보가 아니고, test 쌍이면 정답이 학습에 새어 들어간다"""
    stats: Counter = Counter()
    if fb.empty:
        return fb, stats
    users = set(train.userId) | set(test.userId)
    base = pd.MultiIndex.from_frame(pd.concat([train, test])[["userId", "movieId"]])
    bad_user = ~fb.userId.isin(users)
    bad_movie = ~bad_user & ~fb.movieId.isin(movie_ids)
    in_base = ~bad_user & ~bad_movie & pd.MultiIndex.from_frame(fb[["userId", "movieId"]]).isin(base)
    stats["skip_unknown_user"] = int(bad_user.sum())
    stats["skip_unknown_movie"] = int(bad_movie.sum())
    stats["skip_already_in_base"] = int(in_base.sum())
    return fb[~(bad_user | bad_movie | in_base)].reset_index(drop=True), stats


# ---------- 4. 누적 저장 ----------
def load_store(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return empty_ratings()
    return pd.read_csv(path)[COLUMNS].astype({"userId": "int64", "movieId": "int64", "timestamp": "int64"})


def merge_store(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    both = pd.concat([old, new], ignore_index=True).sort_values("timestamp", kind="stable")
    return both.drop_duplicates(["userId", "movieId"], keep="last").reset_index(drop=True)


# ---------- 5~8. 트리거 · 재학습 · 게이트 · 교체 ----------
def _build_model(name: str, params: dict):
    cls = MODEL_CLASSES[name]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}   # 기록용 params 중 생성자가 받는 것만
    return cls(**{k: v for k, v in params.items() if k in accepted})


def _promote(cand_dir: Path, artifacts_dir: Path, old_version: str) -> None:
    prev = artifacts_dir / "previous" / old_version
    prev.mkdir(parents=True, exist_ok=True)
    for name in ("model.joblib", "metrics.json"):
        shutil.copy2(artifacts_dir / name, prev / name)   # 롤백용 백업
    for name in ("model.joblib", "metrics.json"):
        tmp = artifacts_dir / (name + ".tmp")
        shutil.copy2(cand_dir / name, tmp)
        os.replace(tmp, artifacts_dir / name)             # 읽는 쪽이 반쯤 쓰인 파일을 보지 않게 교체


def run(lines: list[str], train: pd.DataFrame, test: pd.DataFrame, movie_ids: set, *,
        artifacts_dir: Path = config.ARTIFACTS_DIR, store_path: Path = config.FEEDBACK_STORE_PATH,
        min_new: int = config.RETRAIN_MIN_NEW_FEEDBACK, max_regression: float = config.RETRAIN_MAX_RMSE_REGRESSION,
        force: bool = False, dry_run: bool = False) -> dict:
    artifacts_dir = Path(artifacts_dir)
    events, bad_lines = parse_events(lines)
    fb, stats = extract_feedback(events)
    stats["unparsable_lines"] = bad_lines
    fb, fstats = filter_feedback(fb, train, test, movie_ids)
    stats.update(fstats)
    stats["usable_from_logs"] = len(fb)

    store, _ = filter_feedback(merge_store(load_store(store_path), fb), train, test, movie_ids)   # 원본이 바뀌었을 수도 있어 다시 거른다
    if not dry_run:
        Path(store_path).parent.mkdir(parents=True, exist_ok=True)
        store.to_csv(store_path, index=False)

    bundle = joblib.load(artifacts_dir / "model.joblib")
    cur_metrics = json.loads((artifacts_dir / "metrics.json").read_text(encoding="utf-8"))
    used = int(cur_metrics.get("retrain", {}).get("feedback_rows", 0))   # 현재 모델이 이미 학습한 피드백 수
    pending = len(store) - used
    report = {"decision": "skip", "applied": False, "dry_run": dry_run, "log_stats": dict(stats),
              "feedback_store_rows": len(store), "feedback_used_by_current": used, "feedback_pending": pending,
              "current_version": bundle["version"]}

    if len(store) == 0 or (pending < min_new and not force) or (pending <= 0):
        report["reason"] = (f"새 피드백 {pending}건 < 기준 {min_new}건 (--force 로 무시 가능)" if len(store)
                            else "사용할 수 있는 피드백이 없습니다")
        return report

    np.random.seed(config.SEED)
    name, params = cur_metrics["model_name"], cur_metrics.get("params", {})
    merged = pd.concat([train[COLUMNS], store[COLUMNS]], ignore_index=True)
    candidate = _build_model(name, params).fit(merged)
    cur_m, cand_m = evaluate(bundle["model"], test), evaluate(candidate, test)   # 같은 고정 test, 피드백은 들어 있지 않다

    ok = cand_m["rmse"] <= cur_m["rmse"] + max_regression
    now = datetime.now(timezone.utc)
    data_hash = hashlib.sha256(merged.to_csv(index=False).encode("utf-8")).hexdigest()[:12]
    version = f"{name}-{now:%Y%m%d}-{data_hash[:6]}"
    report.update({
        "decision": "promote" if ok else "reject", "new_version": version,
        "current_metrics": {k: cur_m[k] for k in ("rmse", "mae", "ndcg@10")},
        "candidate_metrics": {k: cand_m[k] for k in ("rmse", "mae", "ndcg@10")},
        "rmse_delta": cand_m["rmse"] - cur_m["rmse"], "max_regression": max_regression,
        "train_rows": {"base": len(train), "feedback": len(store)}})
    if dry_run:
        return report

    cand_dir = artifacts_dir / "candidates" / version
    cand_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": candidate, "version": version, "data_hash": data_hash,
                 "parent_version": bundle["version"], "feedback_rows": len(store)}, cand_dir / "model.joblib")
    (cand_dir / "metrics.json").write_text(json.dumps({
        "model_version": version, "model_name": name, "params": params, "metrics": cand_m,
        "data_hash": data_hash, "trained_at": now.isoformat(timespec="seconds"), "seed": config.SEED,
        "relevant_threshold": config.RELEVANT_THRESHOLD,
        "retrain": {"parent_version": bundle["version"], "feedback_rows": len(store),
                    "base_train_rows": len(train), "parent_metrics": cur_m}},
        indent=2, ensure_ascii=False), encoding="utf-8")
    if ok:
        _promote(cand_dir, artifacts_dir, bundle["version"])
        report["applied"] = True
    with open(artifacts_dir / "retrain_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"at": now.isoformat(timespec="seconds"), **report}, ensure_ascii=False) + "\n")
    return report


def print_report(r: dict) -> None:
    print("== 로그 처리 ==")
    for k, v in sorted(r["log_stats"].items()):
        print(f"  {k}: {v}")
    print(f"피드백 누적 {r['feedback_store_rows']}건 / 현재 모델이 학습한 {r['feedback_used_by_current']}건 "
          f"/ 새로 쓸 수 있는 {r['feedback_pending']}건")
    if r["decision"] == "skip":
        print(f"건너뜀: {r['reason']}")
        return
    c, n = r["current_metrics"], r["candidate_metrics"]
    print(f"\n{'':<10}{'RMSE':>9}{'MAE':>9}{'NDCG@10':>10}   (고정 test, 피드백 미포함)")
    print(f"{'현재':<10}{c['rmse']:>9.4f}{c['mae']:>9.4f}{c['ndcg@10']:>10.4f}   {r['current_version']}")
    print(f"{'후보':<10}{n['rmse']:>9.4f}{n['mae']:>9.4f}{n['ndcg@10']:>10.4f}   {r['new_version']}")
    print(f"RMSE 변화 {r['rmse_delta']:+.4f} (허용 악화 {r['max_regression']})")
    verdict = {"promote": "통과 -> 교체", "reject": "실패 -> 교체하지 않음"}[r["decision"]]
    print(f"결과: {verdict}" + (" [dry-run: 저장·교체 안 함]" if r["dry_run"] else ""))
    if r["applied"]:
        print("서비스에 반영하려면 서버(컨테이너)를 다시 시작하세요.")


def main() -> int:
    ap = argparse.ArgumentParser(description="피드백 로그로 재학습하고 지표 비교 후 모델 교체")
    ap.add_argument("--log-file", action="append", type=Path, help="JSON 줄 로그 파일 (여러 번 가능). 기본: logs/requests.jsonl")
    ap.add_argument("--config", help="CloudWatch 설정 JSON (recsys/config.example.json 형식). 주면 CloudWatch 에서 읽는다")
    ap.add_argument("--start", help="--config 와 함께: UTC 시작, 예: 2026-11-01T00:00:00Z")
    ap.add_argument("--end", help="--config 와 함께: UTC 끝")
    ap.add_argument("--min-new", type=int, default=config.RETRAIN_MIN_NEW_FEEDBACK)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.config:
        if not (a.start and a.end):
            ap.error("--config 를 쓰면 --start, --end 가 필요합니다")
        from recsys.archive_logs import parse_utc
        from recsys.cloudwatch_sink import load_config
        lines = lines_from_cloudwatch(load_config(a.config), parse_utc(a.start), parse_utc(a.end))
        print(f"CloudWatch 에서 {len(lines)}줄 읽음")
    else:
        paths = a.log_file or [config.REQUEST_LOG_PATH]
        missing = [str(p) for p in paths if not Path(p).exists()]
        if missing:
            ap.error(f"로그 파일이 없습니다: {missing}")
        lines = read_log_files(paths)
        print(f"로그 파일 {len(paths)}개에서 {len(lines)}줄 읽음")

    train = pd.read_csv(config.PROCESSED_DIR / "train.csv")
    test = pd.read_csv(config.PROCESSED_DIR / "test.csv")
    movie_ids = set(pd.read_csv(config.RAW_DIR / "movies.csv", usecols=["movieId"]).movieId)
    report = run(lines, train, test, movie_ids, min_new=a.min_new, force=a.force, dry_run=a.dry_run)
    print_report(report)
    return 2 if report["decision"] == "reject" else 0


if __name__ == "__main__":
    raise SystemExit(main())
