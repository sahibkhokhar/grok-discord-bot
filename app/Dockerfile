# python base image
FROM python:3.10-slim-bullseye

# set working directory
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg libopus0 ca-certificates git build-essential python3-dev libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy bot code
COPY bot.py .

# run the bot
CMD ["python", "bot.py"]