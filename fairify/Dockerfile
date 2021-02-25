FROM python:3.8-slim-buster

WORKDIR /usr/app
COPY ./requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY ./app.py ./
COPY ./resources ./resources