# E-CUP 2026 Task 3 (user-value) submission image.
# Per .clinerules/02-ecup-rules.md R8: <15 GB compressed, entry point `python -u run.py`.
# Per R4: all dependencies installed at build time, no internet at runtime.

FROM python:3.11-slim AS runtime

# LightGBM manylinux wheels require the GNU OpenMP runtime (libgomp1);
# python:*-slim does not include it, so `import lightgbm` fails with
# "OSError: libgomp.so.1: cannot open shared object file" (R8: must run).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (pinned for reproducibility per R6).
RUN pip install --no-cache-dir uv==0.5.7

WORKDIR /app

# Copy dependency manifests first (Docker layer cache).
COPY pyproject.toml uv.lock ./

# Install dependencies WITHOUT dev group (excludes lightgbm from submission image).
RUN uv sync --frozen --no-dev

# Copy source code.
COPY src/ ./src/
COPY run.py ./

# PYTHONPATH so `python -u run.py` can `import ecup26_user_value.*`.
ENV PYTHONPATH=/app/src
# Put venv binaries first so `python` resolves to the venv interpreter
# (uv sync creates .venv but does NOT modify system PATH).
ENV PATH="/app/.venv/bin:$PATH"

# R8: entry point with the two args expected by ECUP_RULES.md.
ENTRYPOINT ["python", "-u", "run.py"]

# Default args can be overridden by `docker run ... --test_data_path ... --output_path ...`.
CMD ["--test_data_path", "/data/test.csv", "--output_path", "/data/submit.csv"]
