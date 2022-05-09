FROM ubuntu:16.04

RUN apt-get -y update && apt-get install python3 -y
RUN mkdir /sync-folder

COPY flask.py flask.py
COPY static/main.html main.html
COPY static/css/common.css common.css
COPY static/css/log_form.css log_form.css
COPY config/development.py development.py

WORKDIR /sync-folder

ENTRYPOINT ["python3", "flask.py"]
