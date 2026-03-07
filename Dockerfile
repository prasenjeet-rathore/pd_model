# Stage 1: Builder

FROM cgr.dev/chainguard/wolfi-base AS builder

WORKDIR /app



# Install the exact Python version and build tools

# Note: Wolfi uses 'r' (revision) numbers; 'python-3.11=3.11.15-r0' is a common format

USER root

RUN apk update && apk add --no-cache \

    python-3.11=3.11.15-r0 \

    python-3.11-dev \

    build-base \

    libgomp



# Create a virtual environment for portability

RUN python3.11 -m venv /app/venv

ENV PATH="/app/venv/bin:$PATH"



COPY requirements-prod.txt .

RUN pip install --no-cache-dir -r requirements-prod.txt



# Stage 2: Runtime (Final Image)

FROM cgr.dev/chainguard/wolfi-base

WORKDIR /app



# Install only the runtime version of Python 3.11.15 and libgomp

USER root

RUN apk update && apk add --no-cache \

    python-3.11=3.11.15-r0 \

    libgomp



# Copy the pre-built virtual environment

COPY --from=builder /app/venv /app/venv

ENV PATH="/app/venv/bin:$PATH"



# Copy application code

COPY src ./src

COPY app ./app

COPY data/03_final ./data/03_final

COPY models ./models



# Security best practice: Run as a non-privileged user

RUN addgroup -S appgroup && adduser -S appuser -G appgroup

USER appuser



EXPOSE 8000

CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]