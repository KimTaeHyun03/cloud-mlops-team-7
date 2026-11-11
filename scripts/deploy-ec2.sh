#!/usr/bin/env bash
# EC2 에서 최신 코드로 이미지를 빌드하고 컨테이너를 교체한 뒤 /health 의 build_id 로 검증한다
# 파일 위치: D:\ysu_26_2\MLOPS\scripts\deploy-ec2.sh   (EC2 에서는 ~/cloud-mlops-team-7/scripts/deploy-ec2.sh)
#
# 실행 (EC2, 레포 폴더에서):
#   bash scripts/deploy-ec2.sh            # master 를 배포
#   bash scripts/deploy-ec2.sh taehyun    # 다른 브랜치를 시험 배포
#   FORCE=1 bash scripts/deploy-ec2.sh    # 같은 커밋이 떠 있어도 다시 배포
#
# 동작: git pull → (같은 커밋이 이미 떠 있으면 종료) → 학습 → docker build → 컨테이너 교체
#       → /health 의 build_id 가 새 커밋과 같은지 확인 → 실패하면 이전 이미지로 복구
set -euo pipefail

BRANCH="${1:-master}"
NAME="${CONTAINER_NAME:-recsys}"
PORT="${PORT:-8080}"
FORCE="${FORCE:-0}"

cd "$(dirname "$0")/.."

if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo docker"; fi

health_build_id() {   # 실행 중인 서비스의 build_id (없으면 빈 문자열)
  curl -s --max-time 3 "http://localhost:${PORT}/health" 2>/dev/null \
    | python3 -c 'import sys,json; print(json.load(sys.stdin).get("build_id",""))' 2>/dev/null || true
}

echo "== 1. 코드 받기 (브랜치: ${BRANCH})"
git fetch origin "${BRANCH}"
git checkout "${BRANCH}"
git pull --ff-only origin "${BRANCH}"
SHA="$(git rev-parse --short HEAD)"
echo "배포할 커밋: ${SHA}"

RUNNING="$(health_build_id)"
echo "지금 떠 있는 build_id: ${RUNNING:-없음}"
if [ "${RUNNING}" = "${SHA}" ] && [ "${FORCE}" != "1" ]; then
  echo "이미 같은 커밋(${SHA})이 배포되어 있습니다. 중복 배포를 건너뜁니다. (다시 하려면 FORCE=1)"
  exit 0
fi

PREV_IMAGE="$(${DOCKER} inspect --format '{{.Config.Image}}' "${NAME}" 2>/dev/null || true)"
echo "이전 이미지(복구용): ${PREV_IMAGE:-없음}"

echo "== 2. 데이터 준비·학습 (artifacts/, data/processed/ 는 git 에 없어서 매번 만든다)"
${DOCKER} run --rm -v "$PWD":/w -w /w python:3.13-slim sh -c \
  "pip install -q --no-cache-dir -r requirements.txt && python recsys/data_prep.py && python recsys/train.py"

echo "== 3. 이미지 빌드"
${DOCKER} build -t "recsys-api:${SHA}" --build-arg "BUILD_ID=${SHA}" .
${DOCKER} inspect --format '이미지 지문: {{.Id}}' "recsys-api:${SHA}"

echo "== 4. 컨테이너 교체"
${DOCKER} rm -f "${NAME}" >/dev/null 2>&1 || true
${DOCKER} run -d --name "${NAME}" -p "${PORT}:8080" --restart unless-stopped "recsys-api:${SHA}" >/dev/null

echo "== 5. /health 로 build_id 검증 (최대 40초)"
OK=0
for _ in $(seq 1 20); do
  GOT="$(health_build_id)"
  if [ "${GOT}" = "${SHA}" ]; then OK=1; break; fi
  sleep 2
done

if [ "${OK}" = "1" ]; then
  echo "배포 성공: build_id=${GOT}"
  curl -s "http://localhost:${PORT}/health"; echo
  exit 0
fi

echo "배포 실패: build_id 가 ${SHA} 로 바뀌지 않았습니다 (현재: ${GOT:-응답 없음})" >&2
${DOCKER} logs --tail 30 "${NAME}" >&2 || true
${DOCKER} rm -f "${NAME}" >/dev/null 2>&1 || true
if [ -n "${PREV_IMAGE}" ]; then
  echo "이전 이미지(${PREV_IMAGE})로 복구합니다" >&2
  ${DOCKER} run -d --name "${NAME}" -p "${PORT}:8080" --restart unless-stopped "${PREV_IMAGE}" >/dev/null
fi
exit 1
