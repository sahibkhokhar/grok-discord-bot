# python base image
FROM python:3.10-slim-bullseye

# set working directory
WORKDIR /app

# copy requirements file
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy bot code
COPY bot.py .

# run the bot
CMD ["python", "bot.py"]