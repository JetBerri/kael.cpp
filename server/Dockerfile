FROM python:3.12-slim AS build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /source
COPY . .
RUN pip wheel --no-deps --wheel-dir /wheels .

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 kael
COPY --from=build /wheels /wheels
COPY server/requirements.txt /app/requirements.txt
COPY server/app.py /app/app.py
RUN pip install --no-cache-dir /wheels/*.whl -r /app/requirements.txt && rm -rf /wheels
USER kael
WORKDIR /app
EXPOSE 8000
VOLUME /weights
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["python", "app.py", "--data", "/weights", "--host", "0.0.0.0"]
