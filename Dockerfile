FROM python:3.8-slim-buster

WORKDIR /app

RUN pip install opencv-python-headless
RUN pip install numpy
RUN pip install scipy

COPY . .

Entrypoint [ "python3", "main.py"]

LABEL student1="Miłosz Książek"