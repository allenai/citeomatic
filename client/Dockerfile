FROM jupyter/all-spark-notebook

MAINTAINER Hsu Han <hsuo@allenai.org>

#Download Grobid
COPY /Users/hsuo/Downloads/grobid/grobid-grobid-parent-0.4.1 /work/grobid/

#Download Node Deps

#Download Python Deps
COPY ai2/citeomatic /work/ai2/citeomatic/

#Start flask and node
honcho start

#Start Grobid
docker run -it --rm -p 8080:8080 lfoppiano/grobid:0.4.1-SNAPSHOT

#Start citeomatic service
RUN gunicorn -b 0.0.0.0:5000 --timeout=600 --reload --log-level=INFO -w 1 -k sync ai2.citeomatic.gunicorn:app

