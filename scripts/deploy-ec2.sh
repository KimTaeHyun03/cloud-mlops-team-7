#!/usr/bin/env bash
# EC2 에서 최신 코드로 이미지를 빌드하고 컨테이너를 교체한 뒤 /health 의 build_id 로 검증한다
# 파일 위치: D:\ysu_26_2\MLOPS\scripts\deploy-ec2.sh   (EC2 에서는 ~/cloud-mlops-team-7/scripts/deploy-ec2.sh)
#
# 실행 (EC2, 레포 폴더에서):
#   bash scripts/deploy-ec2.sh            # master 를 배포
#   bash scripts/deploy-ec2.sh taehyun    # 다른 브랜치를 시험 배포
#   FORCE=1 bash scripts/deploy-ec2.sh    # 같은 커밋이 떠 있어도 다시 배포
#
# 동작: git pull → (같은 커밋이 이미 떠 있으면 종료) → 학습 → (쌓인 피드백이 충분하면) 재학습
#       → docker build → 컨테이너 교체 → /health 의 build_id 가 새 커밋과 같은지 확인 → 실패하면 이전 이미지로 복구
#
# 서버에 남는 것 (배포해도 사라지지 않는다, 레포에는 올라가지 않는다):
#   ~/recsys-log/logs/requests.jsonl   컨테이너의 /app/logs 를 여기에 연결해 요청·피드백 기록을 보존한다
#   ~/recsys-log/config.json           (선택) CloudWatch 전송 설정
#   ~/recsys-log/aws.env               (선택) AWS 자격증명. EC2 에 IAM 역할이 있으면 필요 없다
#   data/feedback/feedback_ratings.csv 지금까지 모은 (유저, 영화, 실제 평점). retrain.py 가 누적한다
#
# 이 스크립트는 git pull 로 자기 자신이 바뀔 수 있다. bash 는 스크립트를 읽으면서 실행하므로, 본문을 함수로 감싸고
# 마지막 줄에서 호출과 종료를 한 줄로 처리해 실행 도중 파일이 바뀌어도 새 내용을 잘못 읽지 않게 한다.
set -euo pipefail

main() {
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
    return 0
  fi

  PREV_IMAGE="$(${DOCKER} inspect --format '{{.Config.Image}}' "${NAME}" 2>/dev/null || true)"
  echo "이전 이미지(복구용): ${PREV_IMAGE:-없음}"

  # 요청·피드백 기록은 컨테이너 밖(서버 폴더)에 둔다. 컨테이너를 새로 만들어도 남고, 재학습이 여기서 읽는다.
  # 컨테이너는 root 가 아닌 사용자로 돌아서 쓰기 가능하게 열어 둔다.
  LOG_HOST_DIR="${RECSYS_LOG_DIR:-$HOME/recsys-log/logs}"
  mkdir -p "${LOG_HOST_DIR}"
  chmod 777 "${LOG_HOST_DIR}"
  LOG_FILE="${LOG_HOST_DIR}/requests.jsonl"

  echo "== 2. 데이터 준비·학습 (artifacts/, data/processed/ 는 git 에 없어서 매번 만든다)"
  TRAIN_CMD="pip install -q --no-cache-dir -r requirements.txt && python recsys/data_prep.py && python recsys/train.py"
  MOUNTS=(-v "$PWD":/w)
  if [ -s "${LOG_FILE}" ]; then
    echo "쌓인 기록으로 피드백 재학습을 시도합니다: ${LOG_FILE} ($(wc -l < "${LOG_FILE}")줄)"
    # 재학습이 보류(피드백 부족)·거절(RMSE 악화)돼도 배포는 계속한다. 결과 표는 이 로그에 남는다.
    TRAIN_CMD="${TRAIN_CMD} && { python recsys/retrain.py --log-file /logs/requests.jsonl || echo '재학습 보류 또는 거절(종료 코드 '\$?'): 원본 데이터 모델로 배포합니다'; }"
    MOUNTS+=(-v "${LOG_HOST_DIR}":/logs:ro)
  else
    echo "쌓인 기록이 없어 재학습은 건너뜁니다 (${LOG_FILE})"
  fi
  ${DOCKER} run --rm "${MOUNTS[@]}" -w /w python:3.13-slim sh -c "${TRAIN_CMD}"

  echo "== 3. 이미지 빌드"
  ${DOCKER} build -t "recsys-api:${SHA}" --build-arg "BUILD_ID=${SHA}" .
  ${DOCKER} inspect --format '이미지 지문: {{.Id}}' "recsys-api:${SHA}"

  echo "== 4. 컨테이너 교체"
  # CloudWatch 전송 설정은 비밀(자격증명)이 들어갈 수 있어 레포에 두지 않고 서버의 ~/recsys-log/ 에서 읽는다.
  #   ~/recsys-log/config.json  전송 설정 (recsys/config.example.json 형식). 있으면 전송을 켠다
  #   ~/recsys-log/aws.env      (선택) AWS_ACCESS_KEY_ID=... 줄들. EC2 에 IAM 역할이 있으면 필요 없다
  LOG_CFG="${RECSYS_LOG_CONFIG_FILE:-$HOME/recsys-log/config.json}"
  LOG_ENV="${RECSYS_LOG_ENV_FILE:-$HOME/recsys-log/aws.env}"
  RUN_OPTS=(-v "${LOG_HOST_DIR}":/app/logs)
  if [ -f "${LOG_CFG}" ]; then
    echo "CloudWatch 전송 설정 사용: ${LOG_CFG}"
    RUN_OPTS+=(-v "${LOG_CFG}:/run/recsys/config.json:ro" -e RECSYS_LOG_CONFIG=/run/recsys/config.json)
    if [ -f "${LOG_ENV}" ]; then RUN_OPTS+=(--env-file "${LOG_ENV}"); fi
  else
    echo "CloudWatch 전송 설정 없음(${LOG_CFG}): 파일·stdout 로그만 남깁니다"
  fi
  ${DOCKER} rm -f "${NAME}" >/dev/null 2>&1 || true
  ${DOCKER} run -d --name "${NAME}" -p "${PORT}:8080" --restart unless-stopped "${RUN_OPTS[@]}" "recsys-api:${SHA}" >/dev/null

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
    return 0
  fi

  echo "배포 실패: build_id 가 ${SHA} 로 바뀌지 않았습니다 (현재: ${GOT:-응답 없음})" >&2
  ${DOCKER} logs --tail 30 "${NAME}" >&2 || true
  ${DOCKER} rm -f "${NAME}" >/dev/null 2>&1 || true
  if [ -n "${PREV_IMAGE}" ]; then
    echo "이전 이미지(${PREV_IMAGE})로 복구합니다" >&2
    ${DOCKER} run -d --name "${NAME}" -p "${PORT}:8080" --restart unless-stopped "${RUN_OPTS[@]}" "${PREV_IMAGE}" >/dev/null
  fi
  return 1
}

main "$@"; exit $?
