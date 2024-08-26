# Dockerfile for dev container
FROM nvcr.io/nvidia/pytorch:23.03-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Berlin

RUN apt-get update

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ENV PYTHONPATH=/workspaces/psinet/