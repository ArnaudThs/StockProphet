# StockProphet Multi-Ticker Trading System
# Self-contained Docker image for cloud deployment

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire multiticker_refactor folder
COPY . .

# Create necessary directories
RUN mkdir -p saved_models data_cache docs/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Default command: train PPO agent
CMD ["python", "-m", "multiticker_refactor.main_multi", "--mode", "train", "--timesteps", "200000"]

# Alternative commands:
# Evaluate: docker run <image> python -m multiticker_refactor.main_multi --mode evaluate
# Streamlit: docker run -p 8501:8501 <image> streamlit run streamlit_demo/app.py
