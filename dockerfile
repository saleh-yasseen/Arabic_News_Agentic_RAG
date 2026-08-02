FROM python:3.11-slim

WORKDIR /app

ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1

COPY requirements1.txt .
RUN pip install --no-cache-dir -r requirements1.txt

COPY . .
RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]