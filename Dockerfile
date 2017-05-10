FROM s2-research/server-base

RUN mkdir -p /root/.keras

ADD citeomatic/requirements.lock /tmp/requirements.lock
RUN pip install -r /tmp/requirements.lock

ENV PYTHONBUFFERED=1
ENV GROBID_HOST=http://grobid:8080

EXPOSE 5000

RUN pip install gunicorn honcho

WORKDIR /work/
ADD . /work/
RUN pip install -e /work/citeomatic/

CMD honcho -f Procfile.prod start
