# mlflow_server/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
EXPOSE 8080
CMD mlflow server \
    --host 0.0.0.0 \
    --port 8080 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://fpl-modelling-data/mlflow/artifacts/