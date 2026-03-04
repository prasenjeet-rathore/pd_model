FROM python:3.11-slim AS builder
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1

# Install production dependencies to the user site-packages
COPY requirements-prod.txt .
RUN pip install --no-cache-dir --user -r requirements-prod.txt

FROM python:3.11-slim
WORKDIR /app

# 1. FIX: Copy the correct directory for --user installs
COPY --from=builder /root/.local /root/.local

# 2. FIX: Add the new bin directory to the system PATH
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src ./src
COPY app ./app
COPY data/03_final ./data/03_final
COPY models ./models

# 3. FIX: Ensure libgomp1 is available in the final image for scikit-learn/scipy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

EXPOSE 8000
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]