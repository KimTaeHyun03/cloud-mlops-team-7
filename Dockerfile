# 영화 평점 예측 API 이미지
# 파일 위치: D:\ysu_26_2\MLOPS\Dockerfile
#
# 빌드 전에 artifacts/model.joblib, data/processed/ 가 있어야 한다 (python recsys/data_prep.py, python recsys/train.py).
# 빌드 (PowerShell, 프로젝트 루트에서):
#   docker build -t recsys-api:local --build-arg BUILD_ID=$(git rev-parse --short HEAD) .
# 실행:
#   docker run --rm -p 8080:8080 --name a9-recsys recsys-api:local
# 확인:
#   curl http://localhost:8080/health
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 의존성 먼저 (코드만 바뀌면 이 층은 캐시를 재사용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 + 모델 + 실행에 필요한 데이터 (.dockerignore 가 나머지를 걸러낸다)
COPY recsys ./recsys
COPY artifacts ./artifacts
COPY data/processed ./data/processed
COPY data/ml-latest-small/movies.csv data/ml-latest-small/links.csv ./data/ml-latest-small/
COPY top_rated_movies_utf8.csv ./

# 배포한 커밋을 이미지에서 확인할 수 있게 남긴다
ARG BUILD_ID=unknown
ENV BUILD_ID=${BUILD_ID}

# root 가 아닌 사용자로 실행 (요청 기록을 쓰는 logs/ 만 쓰기 허용)
RUN useradd --system --no-create-home app \
    && mkdir -p /app/logs \
    && chown app:app /app/logs
USER app

EXPOSE 8080

# 컨테이너 안에서 /health 응답을 주기적으로 확인 (slim 이미지에는 curl 이 없어 python 사용)
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=2).status == 200 else 1)"

CMD ["uvicorn", "recsys.api:app", "--host", "0.0.0.0", "--port", "8080"]
