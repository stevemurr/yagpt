FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
COPY yagpt/ yagpt/
COPY scripts/ scripts/
COPY web/ web/
COPY configs/ configs/

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 && \
    pip install --no-cache-dir ".[web,data]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "web.backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
