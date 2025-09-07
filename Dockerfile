# Dockerfile for Question_mk
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exclude .env from image
RUN rm -f .env

EXPOSE 8000

CMD ["python", "main.py"]
