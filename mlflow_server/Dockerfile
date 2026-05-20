FROM python:3.11-slim
WORKDIR /app

# Install only what mlflow server actually needs
RUN pip install --no-cache-dir \
    mlflow==2.13.0 \
    boto3 \
    pymysql

EXPOSE 8080
CMD mlflow server \
    --host 0.0.0.0 \
    --port 8080 \
    --backend-store-uri ${DATABASE_URL} \
    --default-artifact-root s3://fpl-modelling-data/mlflow/artifacts/