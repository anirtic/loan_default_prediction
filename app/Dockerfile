FROM python:3.9.13
WORKDIR /app
COPY main.py ./
COPY requirements.txt ./
COPY config.py ./
COPY functions.py ./
COPY data_extraction/ ./data_extraction/
RUN pip install --no-cache-dir --upgrade -r requirements.txt
EXPOSE 8000
RUN useradd -m myuser
USER myuser
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


