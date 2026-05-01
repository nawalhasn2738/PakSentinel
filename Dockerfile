FROM python:3.12-slim

WORKDIR /app

# Install system dependencies for psycopg2, nltk, and other packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       gcc \
       libxml2-dev \
       libxslt1-dev \
       libffi-dev \
       curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python -m nltk.downloader punkt stopwords wordnet omw-1.4

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
