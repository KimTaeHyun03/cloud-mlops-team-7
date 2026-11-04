#!/usr/bin/env bash
# 태그로 EC2 를 찾아 켜고, 없으면 새로 만들어 git clone·Docker 설정까지 해 둔다
# 파일 위치: D:\ysu_26_2\MLOPS\scripts\ensure-ec2.sh   (GitHub Actions 와 CloudShell 에서 AWS CLI 로 실행)
#
# 실행 (AWS 자격이 설정된 곳, 레포 루트에서):
#   TAG_VALUE=team7-recsys SECURITY_GROUP_ID=sg-xxxx bash scripts/ensure-ec2.sh
#
# 환경변수
#   TAG_VALUE           (필수) 서버에 붙은 태그 값
#   TAG_KEY             태그 키, 기본 Name
#   SECURITY_GROUP_ID   (서버를 새로 만들 때 필수) 22·8080 포트가 열린 보안 그룹
#   INSTANCE_TYPE       기본 t3.micro   ※ 기존 서버와 같은 타입으로 맞출 것
#   AMI_ID              기본: Amazon Linux 2023 최신 x86_64 (SSM 공개 파라미터)
#   SUBNET_ID, KEY_NAME, IAM_PROFILE   선택
#   VOLUME_GB           기본 20
#   SSH_PUBLIC_KEY      새 서버의 ec2-user authorized_keys 에 넣을 공개 키 (Actions 의 접속 키와 짝)
#   REPO_URL, BRANCH    기본 이 레포, master
# 결과: 화면에 서버 상태·주소를 출력하고, $GITHUB_OUTPUT 이 있으면 host=, created=, instance_id= 를 기록한다
set -euo pipefail

TAG_KEY="${TAG_KEY:-Name}"
TAG_VALUE="${TAG_VALUE:?TAG_VALUE 가 필요합니다}"
INSTANCE_TYPE="${INSTANCE_TYPE:-t3.micro}"
VOLUME_GB="${VOLUME_GB:-20}"
REPO_URL="${REPO_URL:-https://github.com/KimTaeHyun03/cloud-mlops-team-7.git}"
BRANCH="${BRANCH:-master}"
CREATED=false

q() { aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --query "$1" --output text; }

# 삭제된(terminated) 인스턴스는 한동안 목록에 남으므로 상태로 걸러낸다
IDS="$(aws ec2 describe-instances \
  --filters "Name=tag:${TAG_KEY},Values=${TAG_VALUE}" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text)"
COUNT="$(wc -w <<<"$IDS")"

if [ "$COUNT" -gt 1 ]; then
  echo "태그 ${TAG_KEY}=${TAG_VALUE} 인 서버가 ${COUNT}대입니다: ${IDS}. 하나만 남기세요" >&2
  exit 1
fi

if [ "$COUNT" -eq 0 ]; then
  echo "태그 ${TAG_KEY}=${TAG_VALUE} 인 서버가 없어서 새로 만듭니다"
  : "${SECURITY_GROUP_ID:?서버를 새로 만들려면 SECURITY_GROUP_ID(22·8080 열린 보안 그룹)가 필요합니다}"

  if [ -z "${AMI_ID:-}" ]; then
    AMI_ID="$(aws ssm get-parameter \
      --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
      --query Parameter.Value --output text)"
  fi
  echo "AMI: ${AMI_ID}, 타입: ${INSTANCE_TYPE}, 보안 그룹: ${SECURITY_GROUP_ID}"

  # 첫 부팅 때 한 번 실행되는 초기 설정: Docker·git 설치, 부팅 시 Docker 자동 시작, 레포 받기
  USER_DATA="$(mktemp)"
  {
    echo '#!/bin/bash'
    echo 'set -eux'
    echo 'dnf install -y docker git'
    echo 'systemctl enable --now docker'
    echo 'usermod -aG docker ec2-user'
    echo 'install -d -m 700 -o ec2-user -g ec2-user /home/ec2-user/.ssh'
    if [ -n "${SSH_PUBLIC_KEY:-}" ]; then
      printf 'echo %q >> /home/ec2-user/.ssh/authorized_keys\n' "$SSH_PUBLIC_KEY"
      echo 'chown ec2-user:ec2-user /home/ec2-user/.ssh/authorized_keys'
      echo 'chmod 600 /home/ec2-user/.ssh/authorized_keys'
    fi
    printf 'sudo -u ec2-user git clone --branch %q %q /home/ec2-user/cloud-mlops-team-7\n' "$BRANCH" "$REPO_URL"
    echo 'sudo -u ec2-user touch /home/ec2-user/.bootstrap-done'
  } > "$USER_DATA"

  ARGS=(--image-id "$AMI_ID" --instance-type "$INSTANCE_TYPE"
        --security-group-ids "$SECURITY_GROUP_ID"
        --user-data "file://${USER_DATA}"
        --block-device-mappings "DeviceName=/dev/xvda,Ebs={VolumeSize=${VOLUME_GB},VolumeType=gp3}"
        --metadata-options "HttpTokens=required"
        --tag-specifications "ResourceType=instance,Tags=[{Key=${TAG_KEY},Value=${TAG_VALUE}}]"
        --count 1 --query 'Instances[0].InstanceId' --output text)
  [ -n "${SUBNET_ID:-}" ]   && ARGS+=(--subnet-id "$SUBNET_ID")
  [ -n "${KEY_NAME:-}" ]    && ARGS+=(--key-name "$KEY_NAME")
  [ -n "${IAM_PROFILE:-}" ] && ARGS+=(--iam-instance-profile "Name=${IAM_PROFILE}")

  INSTANCE_ID="$(aws ec2 run-instances "${ARGS[@]}")"
  rm -f "$USER_DATA"
  CREATED=true
  echo "새 서버 생성: ${INSTANCE_ID}"
else
  INSTANCE_ID="$IDS"
  STATE="$(q 'Reservations[0].Instances[0].State.Name')"
  echo "서버 ${INSTANCE_ID} 상태: ${STATE}"
  if [ "$STATE" = "stopping" ]; then
    aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
    STATE=stopped
  fi
  if [ "$STATE" = "stopped" ]; then
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" >/dev/null
  fi
fi

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"

# 켠 뒤의 현재 퍼블릭 주소 (껐다 켜거나 새로 만들면 바뀐다)
HOST="$(q 'Reservations[0].Instances[0].PublicDnsName')"
if [ -z "$HOST" ] || [ "$HOST" = "None" ]; then
  HOST="$(q 'Reservations[0].Instances[0].PublicIpAddress')"
fi
echo "서버 최종 상태: $(q 'Reservations[0].Instances[0].State.Name'), 주소: ${HOST}, 새로 만듦: ${CREATED}"

if [ -n "${GITHUB_OUTPUT:-}" ]; then
  {
    echo "host=${HOST}"
    echo "created=${CREATED}"
    echo "instance_id=${INSTANCE_ID}"
  } >> "$GITHUB_OUTPUT"
fi
