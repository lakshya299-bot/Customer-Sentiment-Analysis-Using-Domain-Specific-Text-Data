
FROM --platform=linux/amd64 python:3.9-slim


WORKDIR /app


ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .



RUN pip install --no-cache-dir -r requirements.txt


COPY app.py .



COPY model_quantized_onnx ./model_quantized_onnx


EXPOSE 8501



CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
