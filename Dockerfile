FROM python:3.10-slim

# Install libGL untuk OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "app:app"]