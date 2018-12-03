FROM python:3.6-slim

ENV HOME=/root

WORKDIR $HOME

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential

COPY requirements.txt $HOME/
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . $HOME/

CMD ["pytest", "--color=yes", "-s", "tests.py"]
