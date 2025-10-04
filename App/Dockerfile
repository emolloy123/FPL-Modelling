# ---- Base image ----
FROM python:3.12.6-slim

# ---- Set environment variables ----
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

# ---- Install system dependencies ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Poetry ----
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# ---- Set working directory ----
WORKDIR /app

# ---- Copy Poetry project files from fpl-modelling ----
COPY ../fpl-modelling/pyproject.toml ../fpl-modelling/poetry.lock ./fpl-modelling/

# ---- Install project dependencies ----
WORKDIR /app/fpl-modelling
RUN poetry install --no-root --without dev

# ---- Copy the rest of the code ----
COPY ../fpl-modelling ./
COPY ../App/streamlit_app.py /app/

# ---- Expose Streamlit port ----
EXPOSE 8501

# ---- Health check ----
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---- Run Streamlit ----
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
