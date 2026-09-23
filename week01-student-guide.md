# 1주차 학생 안내 — 팀을 만들고 AWS에 첫 파일 올리기

수업은 50분씩 4교시이며, 교시 사이에 10분씩 쉽니다. 실제 수업 활동 시간은 200분입니다.

## 오늘 남길 결과

- 압축을 푼 `team-start-kit` 폴더
- 3~4명 팀과 팀 규칙, 1~4주 역할
- 프로젝트 후보 두 개
- 서울 리전의 비공개 S3 버킷과 `sample-bike.csv`
- CloudShell에서 실행한 `aws s3 ls` 결과
- 남은 수업의 난이도를 정하는 Cloud·ML·Ops 기초 진단 24문항 응답

## 파일 받는 곳

수업 중 필요한 파일은 현재 장표 아래의 **첨부파일**에서 받습니다.

- 42번: 팀 시작 파일과 이 학생 안내서
- 60번: 프로젝트 예시 주제 안내
- 75번: AWS 로그인·S3 실습 안내
- 89번: `sample-bike.csv`
- 94번: 팀 시작 파일과 학생 안내서 다시 받기

ZIP 파일은 먼저 압축을 풀고, 압축파일이 아니라 풀린 폴더에서 작업합니다.

## 1교시

가격 계산 화면은 정해진 식을 사용하는 예제입니다. 아직 머신러닝 모델은 아닙니다. 이번 학기에는 화면, API, 모델, 배포, 로그를 차례로 연결합니다. 팀 시작 파일을 받은 뒤 `README.md`를 먼저 읽습니다.

10번 장표에서 Cloud·ML·Ops 기초 용어 24문항에 답합니다. 모르는 문항은 추측하지 말고 `모르겠다`를 선택합니다. 진단 정답 수는 성적에 반영하지 않습니다.

11번 장표에서는 이름과 개별 답안 없이 각 선택지의 비율, 미응답 비율, 정답과 해설을 확인합니다. 자신이 헷갈린 문항을 중심으로 읽습니다.

## 2교시

1. `SKILL_PASSPORT.md`에 해 본 일과 맡고 싶은 역할을 적습니다.
2. 서로 다른 학생 세 명과 작업 방식과 희망 역할을 30초씩 이야기합니다.
3. 3~4명으로 팀을 만듭니다.
4. `TEAM_CHARTER.md`에 연락·회의·마감·갈등 처리 규칙을 적습니다.
5. `ROLE_ROTATION.csv`에 1~4주 역할을 정합니다.
6. 화면 오른쪽 위 **팀 구성**에서 팀 코드를 이용해 참여하고, 전원이 확정에 동의합니다.

## 3교시

예시 주제 8개를 살펴보고, 각자 후보 두 개를 개인 메모에 고릅니다. 팀에서 후보 세 개를 같은 기준으로 비교한 뒤 1순위와 예비 후보를 남깁니다. 3주차에 실제 데이터를 열어 본 뒤 최종 결정합니다.

> `[사용자]`가 `[입력]`을 넣으면 `[예측 결과]`를 받고, `[화면]`에서 다음 행동을 결정한다.

## 4교시

1. 학교에서 받은 AWS access portal로 로그인합니다.
2. 수업용 계정·역할을 선택합니다.
3. 리전을 `아시아 태평양(서울) ap-northeast-2`로 맞춥니다.
4. `S3 → Buckets → Create bucket`으로 이동합니다.
5. `mlops-2026-<학번>-<임의문자>` 형식으로 버킷 이름을 정합니다.
6. `Bucket owner enforced`와 `Block all public access`를 유지합니다.
7. 89번 장표에서 `sample-bike.csv`를 받고 버킷에 올립니다.
8. CloudShell에서 아래 명령을 실행합니다.

```bash
aws s3 ls
aws s3 ls s3://내-버킷-이름/
aws s3 cp s3://내-버킷-이름/sample-bike.csv s3://내-버킷-이름/raw/sample-bike.csv
aws s3 ls s3://내-버킷-이름/raw/
```


## 안전하게 마치기

- 오늘 만든 버킷은 다음 주에 쓰므로 삭제하지 않습니다.
- 퍼블릭 액세스 차단은 그대로 둡니다.
- 계정 번호·액세스 키·비밀번호가 문서나 화면 캡처에 없는지 확인합니다.

## AWS 공식 문서

- [IAM Identity Center 사용자 포털](https://docs.aws.amazon.com/singlesignon/latest/userguide/using-the-portal.html)
- [S3 버킷 만들기](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)
- [S3 퍼블릭 액세스 차단](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html)
- [AWS CloudShell](https://docs.aws.amazon.com/cloudshell/latest/userguide/welcome.html)
