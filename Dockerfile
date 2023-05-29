FROM python:3.9-slim-buster

WORKDIR /app

ADD . /app

RUN apt update && apt install -y \
    libgl1-mesa-dev \
    libglib2.0-0

RUN chmod +x /app/setup.sh /app/compress.sh /app/decompress.sh /app/main.sh

RUN /app/setup.sh

ENTRYPOINT ["/app/main.sh"]
