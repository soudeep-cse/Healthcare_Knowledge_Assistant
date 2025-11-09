FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-jpn \
    libtesseract-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set tesseract environment variables
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/
ENV PATH="/usr/bin:${PATH}"

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p faiss_index docs

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]