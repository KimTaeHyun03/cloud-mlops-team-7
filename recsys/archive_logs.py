r"""
기간을 정해 CloudWatch 기록을 찾아 S3 에 보관하고, 다시 읽어 건수·지문을 대조한다
파일 위치: D:\ysu_26_2\MLOPS\recsys\archive_logs.py

실행 명령 (CloudShell, 프로젝트 루트에서. 시각은 UTC):
  python -m recsys.archive_logs --config config.json --start 2026-10-28T00:00:00Z --end 2026-10-28T12:00:00Z
  python -m recsys.archive_logs --config config.json --start ... --end ... --dry-run   # S3 에 쓰지 않고 조회·지문만 확인

config.json 에 s3_bucket(수업에서 허용한 버킷), s3_prefix(새 경로)가 필요하다.
S3 에는 data.jsonl(기록)과 query.json(조회 조건·건수·sha256)을 쓰고, 다시 내려받아 원본과 대조한다.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recsys.cloudwatch_sink import check_account, load_config, make_client


def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)


def fetch_messages(logs, group: str, stream: str, start: datetime, end: datetime) -> list[str]:
    kwargs = dict(logGroupName=group, logStreamNames=[stream],
                  startTime=int(start.timestamp() * 1000), endTime=int(end.timestamp() * 1000))
    events = []
    while True:
        page = logs.filter_log_events(**kwargs)
        events += page["events"]
        if not page.get("nextToken"):
            break
        kwargs["nextToken"] = page["nextToken"]
    events.sort(key=lambda e: (e["timestamp"], e["eventId"]))
    return [e["message"] for e in events]


def fingerprint(lines: list[str]) -> str:
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def archive(cfg: dict, start: datetime, end: datetime, dry_run: bool = False, logs=None, s3=None) -> dict:
    logs = logs or make_client(cfg, "logs")
    lines = fetch_messages(logs, cfg["log_group"], cfg["log_stream"], start, end)
    query = {"region": cfg["region"], "log_group": cfg["log_group"], "log_stream": cfg["log_stream"],
             "start_utc": start.isoformat(), "end_utc": end.isoformat(), "timezone": "UTC",
             "count": len(lines), "sha256": fingerprint(lines),
             "archived_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if dry_run:
        return {"query": query, "reread": None}

    for key in ("s3_bucket", "s3_prefix"):
        if not cfg.get(key):
            raise ValueError(f"config 에 {key} 가 필요합니다 (수업에서 허용한 버킷·새 경로)")
    s3 = s3 or make_client(cfg, "s3")
    base = f"{cfg['s3_prefix'].strip('/')}/{start:%Y%m%dT%H%M%SZ}_{end:%Y%m%dT%H%M%SZ}"
    body = ("\n".join(lines) + "\n").encode("utf-8")
    s3.put_object(Bucket=cfg["s3_bucket"], Key=f"{base}/data.jsonl", Body=body, ContentType="application/x-ndjson")
    s3.put_object(Bucket=cfg["s3_bucket"], Key=f"{base}/query.json",
                  Body=json.dumps(query, ensure_ascii=False, indent=2).encode("utf-8"), ContentType="application/json")

    back = s3.get_object(Bucket=cfg["s3_bucket"], Key=f"{base}/data.jsonl")["Body"].read().decode("utf-8")
    back_lines = back.splitlines()
    reread = {"s3_uri": f"s3://{cfg['s3_bucket']}/{base}/data.jsonl", "count": len(back_lines),
              "sha256": fingerprint(back_lines),
              "match": len(back_lines) == query["count"] and fingerprint(back_lines) == query["sha256"]}
    return {"query": query, "reread": reread}


def main() -> int:
    ap = argparse.ArgumentParser(description="CloudWatch 기록을 기간별로 S3 에 보관하고 다시 읽어 대조")
    ap.add_argument("--config", required=True)
    ap.add_argument("--start", required=True, help="UTC, 예: 2026-10-28T00:00:00Z")
    ap.add_argument("--end", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_config(a.config)
    check_account(cfg)
    result = archive(cfg, parse_utc(a.start), parse_utc(a.end), a.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["reread"] and not result["reread"]["match"]:
        print("다시 읽은 자료가 원본과 다릅니다", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
