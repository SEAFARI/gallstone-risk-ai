FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Streamlit runs on port 8501 by default
EXPOSE 8501

# Important: Bind to 0.0.0.0 so Hugging Face can "see" the app
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
