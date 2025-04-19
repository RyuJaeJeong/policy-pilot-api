FROM python:3.12-slim
LABEL author="DX사업팀 류재정 프로"
LABEL description="FAST API를 위한 이미지 빌드"
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install uv
WORKDIR /api
COPY . .
RUN uv venv && uv sync
EXPOSE 8000
ENTRYPOINT [".venv/bin/python", "run.py"]
