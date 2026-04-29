# Multi-stage Docker build for Virtual ICU
FROM python:3.10-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel && pip install -r requirements.txt

FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN useradd -m -u 1000 appuser
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE 10000
CMD ["sh", "-c", "streamlit run app_v2.py --server.port=10000 --server.address=0.0.0.0 --server.headless=true"]
