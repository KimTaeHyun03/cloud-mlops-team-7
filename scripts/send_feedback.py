r"""
명령어로 예측 요청 + 실제 별점(피드백)을 서버에 넣는다. 재학습용 로그를 쌓을 때 쓴다 (화면 없이).
파일 위치: D:\ysu_26_2\MLOPS\scripts\send_feedback.py

별점은 사람이 직접 채운 CSV 에서만 읽는다 (임의로 만들지 않는다. 가짜 평점으로 재학습하면 개선 수치가 의미가 없다).

실행 명령 (PowerShell, 프로젝트 루트에서):
  # 1) 후보 (유저, 영화) 목록 템플릿 만들기: 유저 21~25 가 각각 평가해 볼 인기 영화 10편 (이미 평점 있는 영화는 제외)
  python scripts/send_feedback.py make-template --users 21-25 --per-user 10 --out my_ratings.csv
  # 2) my_ratings.csv 를 열어 rating 칸에 본 영화의 실제 별점(0.5~5.0, 0.5 단위)을 적는다. 안 본 영화는 비워 둔다
  # 3) 서버로 보내기 (먼저 --dry-run 으로 확인)
  python scripts/send_feedback.py send --base-url http://<서버주소>:8080 --csv my_ratings.csv --dry-run
  python scripts/send_feedback.py send --base-url http://<서버주소>:8080 --csv my_ratings.csv

CSV 형식: user_id,movie_id,rating  (title 열은 있어도 되고 무시한다)
동작: 행마다 POST /predict 로 request_id 를 받고, 바로 POST /feedback 으로 별점을 보낸다 (request_id 는 서버 메모리에 있어서
      예측 직후 보내야 한다. 서버를 재시작하면 그 전 요청의 별점은 404 로 거부된다).
외부 라이브러리 없이 표준 라이브러리만 쓰는 send 와 달리, make-template 은 pandas 와 data/processed 가 필요하다.
"""
import argparse
import csv
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def valid_rating(x: float) -> bool:
    return 0.5 <= x <= 5.0 and (x * 2) == int(x * 2)


def parse_range(text: str) -> list[int]:
    """'21-25' -> [21..25], '3,7,9' -> [3,7,9], '5' -> [5]"""
    ids: list[int] = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            ids += list(range(int(a), int(b) + 1))
        else:
            ids.append(int(part))
    return ids


# ---------- make-template ----------
def make_template(users: list[int], per_user: int, out: Path, pool: int) -> int:
    import pandas as pd
    proc = ROOT / "data" / "processed"
    train, test = pd.read_csv(proc / "train.csv"), pd.read_csv(proc / "test.csv")
    movies = pd.read_csv(ROOT / "data" / "ml-latest-small" / "movies.csv").set_index("movieId")["title"]
    seen = pd.concat([train, test])[["userId", "movieId"]]
    popular = train.groupby("movieId").size().sort_values(ascending=False).head(pool).index.tolist()   # 많이 본 영화 = 팀원이 봤을 확률이 높다
    rows = []
    for u in users:
        done = set(seen.loc[seen.userId == u, "movieId"])
        if u not in set(seen.userId):
            print(f"건너뜀: 유저 {u} 는 학습 데이터에 없습니다 (1~610 만 가능)", file=sys.stderr)
            continue
        mine = [m for m in popular if m not in done][:per_user]
        rows += [(u, m, movies.get(m, ""), "") for m in mine]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: 엑셀에서 한글이 안 깨진다
        w = csv.writer(f)
        w.writerow(["user_id", "movie_id", "title", "rating"])
        w.writerows(rows)
    print(f"{out} 에 {len(rows)}행 작성. rating 칸에 본 영화의 실제 별점(0.5~5.0, 0.5 단위)을 채우고, 안 본 영화는 비워 두세요")
    return 0


# ---------- send ----------
def post(base: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(base.rstrip("/") + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def read_rows(csv_path: Path) -> tuple[list[tuple[int, int, float]], list[str]]:
    rows, problems = [], []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for n, r in enumerate(csv.DictReader(f), start=2):
            raw = (r.get("rating") or "").strip()
            if not raw:
                continue   # 평가하지 않은 행
            try:
                u, m, x = int(r["user_id"]), int(r["movie_id"]), float(raw)
            except (ValueError, KeyError, TypeError):
                problems.append(f"{n}행: 숫자로 읽을 수 없음 ({dict(r)})")
                continue
            if u < 1 or m < 1 or not valid_rating(x):
                problems.append(f"{n}행: 값이 올바르지 않음 (유저 {u}, 영화 {m}, 별점 {x}) — 별점은 0.5~5.0, 0.5 단위")
                continue
            rows.append((u, m, x))
    return rows, problems


def send(base: str, csv_path: Path, dry_run: bool) -> int:
    rows, problems = read_rows(csv_path)
    for p in problems:
        print("건너뜀:", p, file=sys.stderr)
    dup = len(rows) - len(set((u, m) for u, m, _ in rows))
    if dup:
        print(f"주의: 같은 (유저, 영화)가 {dup}건 겹칩니다. 서버는 가장 늦게 보낸 것 하나만 학습에 씁니다", file=sys.stderr)
    print(f"보낼 행 {len(rows)}건 (유저 {len({u for u, _, _ in rows})}명)")
    if dry_run:
        print("[dry-run] 서버로 보내지 않았습니다")
        return 0

    ok, errs, abs_err = 0, [], []
    for u, m, x in rows:
        try:
            p = post(base, "/predict", {"user_id": u, "movie_id": m})
            f = post(base, "/feedback", {"request_id": p["request_id"], "movie_id": m, "rating": x})
            ok += 1
            abs_err.append(abs(f["predicted_rating"] - x))
        except urllib.error.HTTPError as e:
            errs.append(f"유저 {u} 영화 {m}: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:120]}")
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            errs.append(f"유저 {u} 영화 {m}: {type(e).__name__}: {e}")
    print(f"성공 {ok}건 / 실패 {len(errs)}건")
    if abs_err:
        print(f"보낸 별점과 서버 예측의 평균 절대 오차: {sum(abs_err) / len(abs_err):.3f}")
    for e in errs[:10]:
        print("실패:", e, file=sys.stderr)
    return 1 if errs else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="재학습용 피드백을 명령어로 서버에 넣는다")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("make-template", help="평점을 채울 후보 (유저, 영화) 목록 CSV 만들기")
    t.add_argument("--users", required=True, help="예: 21-25 또는 3,7,9")
    t.add_argument("--per-user", type=int, default=10)
    t.add_argument("--pool", type=int, default=300, help="후보로 삼을 인기 영화 수 (기본 300)")
    t.add_argument("--out", type=Path, default=Path("my_ratings.csv"))

    s = sub.add_parser("send", help="CSV 의 별점을 서버로 보내기")
    s.add_argument("--base-url", required=True, help="예: http://<서버주소>:8080")
    s.add_argument("--csv", required=True, type=Path)
    s.add_argument("--dry-run", action="store_true")

    a = ap.parse_args()
    if a.cmd == "make-template":
        return make_template(parse_range(a.users), a.per_user, a.out, a.pool)
    return send(a.base_url, a.csv, a.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
