# =============== Base Image ===============
FROM python:3.11-slim

# =============== System Dependencies ===============
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# =============== Working Directory ===============
WORKDIR /app

# =============== Install Python Packages ===============
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =============== Copy App Code ===============
COPY . .

# =============== Streamlit Settings ===============
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV PORT=8501

EXPOSE 8501

# =============== Start App ===============
CMD ["streamlit", "run", "src/app_streamlit.py", "--server.address=0.0.0.0"]
