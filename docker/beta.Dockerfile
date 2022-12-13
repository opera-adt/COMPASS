# Dockerfile for beta release

FROM ubuntu:22.04

LABEL author="OPERA ADT" \
      description="s1 cslc 0.2.0 beta release" \
      version="beta"

RUN apt-get -y update &&\
    apt-get -y install curl git &&\
    adduser --disabled-password compass_user

USER compass_user 

ENV CONDA_PREFIX=/home/compass_user/miniconda3

WORKDIR /home/compass_user

RUN mkdir -p /home/compass_user/OPERA &&\
    cd /home/compass_user/OPERA &&\
    curl -sSL https://github.com/seongsujeong/COMPASS/archive/refs/tags/v0.2.0-beta.tar.gz -o compass_src.tar.gz &&\
    tar -xvf compass_src.tar.gz &&\
    ln -s COMPASS-0.2.0-beta COMPASS &&\
    rm compass_src.tar.gz


RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniconda.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN ${CONDA_PREFIX}/bin/conda init bash

RUN conda create --name "COMPASS" --file /home/compass_user/OPERA/COMPASS/docker/specifile.txt

SHELL ["conda", "run", "-n", "COMPASS", "/bin/bash", "-c"]

RUN echo "Installing OPERA s1-reader" &&\
    cd ${HOME}/OPERA &&\
    curl -sSL https://github.com/seongsujeong/s1-reader/archive/refs/tags/v0.1.4-beta.temp.tar.gz -o s1_reader_src.tar.gz &&\
    tar -xvf s1_reader_src.tar.gz &&\
    ln -s s1-reader-0.1.4-beta.temp s1-reader &&\
    rm s1_reader_src.tar.gz &&\
    python -m pip install ./s1-reader &&\
    echo "Installing OPERA COMPASS" &&\
    python -m pip install ./COMPASS &&\
    echo "conda activate COMPASS" >> /home/compass_user/.bashrc

WORKDIR /home/compass_user/scratch

ENTRYPOINT ["conda", "run", "-n", "COMPASS","s1_cslc.py"]
