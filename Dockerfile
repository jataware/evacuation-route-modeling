FROM --platform=linux/amd64 python:3.8.12


RUN pip install --upgrade pip
RUN apt-get update; apt-get install ffmpeg libsm6 libxext6  -y

COPY . /
WORKDIR /
RUN pip install -r requirements.txt

WORKDIR /Ensemble_Attraction_Routing
RUN ./setup.sh

CMD ["python", "Ensemble-Attraction-Routing.py", "--config_file", "config.json.ensemble"]