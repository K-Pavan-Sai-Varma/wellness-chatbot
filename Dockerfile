
# ⿡ Base image – lightweight and stable Python
FROM python:3.11-slim

# ⿢ Set working directory inside container
WORKDIR /app

# ⿣ Copy dependency list first (for caching)
COPY requirements.txt .

# ⿤ Install dependencies safely (with pip upgrade + sentencepiece)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir sentencepiece \
    && pip install --no-cache-dir -r requirements.txt

# ⿥ Copy your project files
COPY . .

# ⿦ Expose Flask app port
EXPOSE 5000

# ⿧ Command to start your Flask app
CMD ["python","./app.py"]
