# python base image
FROM python:3.10-slim-bullseye

# set working directory
WORKDIR /app

ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       git build-essential ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy bot code and local tools
COPY bot.py local_tools.py .

# run the bot
CMD ["python", "-u", "bot.py"]
