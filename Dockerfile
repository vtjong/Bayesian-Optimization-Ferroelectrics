# Local interactive dashboard for the HZO Bayesian-optimization project.
# Build/run via scripts/run_dashboard.sh docker  (pass MP_API_KEY for the structures tab).
FROM python:3.11-slim

WORKDIR /app

# build-essential covers any source builds in the scientific stack (pymatgen, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching.
COPY requirements.txt requirements-dashboard.txt requirements-viz.txt ./
RUN pip install --no-cache-dir \
    -r requirements.txt \
    -r requirements-dashboard.txt \
    -r requirements-viz.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8501

# MP_API_KEY is supplied at runtime (docker run -e MP_API_KEY=...).
CMD ["streamlit", "run", "src/dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
