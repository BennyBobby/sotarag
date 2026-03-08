FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache
COPY . .
EXPOSE 8501
ENV PYTHONPATH=/app
CMD ["uv", "run", "streamlit", "run", "src/ui/app.py", "--server.address=0.0.0.0", "--server.port=8501"]