##### General instruction: ######
#docker build -f ./Dockerfile -t opera-adt/compass:latest . #To build the docker image
#to run docker container in interactive mode: docker run -it --name COMPASS opera-adt/compass:latest /bin/bash

FROM ubuntu:22.04

LABEL author="OPERA ADT" \
      description="s1 cslc 0.1.0 IF release" \
      version="0.1.0_IF"

RUN apt-get -y update &&\
    apt-get -y install curl git &&\
    adduser --disabled-password compass_user &&\
    mkdir -p /home/compass_user/OPERA/s1-reader &&\
    mkdir -p /home/compass_user/OPERA/OPERA

#COPY docker/specifile.txt /home/compass_user/
COPY specifile.txt /home/compass_user/
COPY s1-reader /home/compass_user/OPERA/s1-reader
COPY COMPASS /home/compass_user/OPERA/COMPASS

RUN chown -R compass_user:compass_user /home/compass_user/*

USER compass_user 

WORKDIR /home/compass_user

#Install miniconda
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniconda.sh

ENV CONDA_PREFIX=/home/compass_user/miniconda3
ENV PATH="${CONDA_PREFIX}/bin:${PATH}"

RUN conda create --name "COMPASS" --file specifile.txt &&\
    $CONDA_PREFIX/bin/conda init bash &&\
    echo "conda activate COMPASS" >> /home/compass_user/.bashrc &&\
    mkdir /home/compass_user/scratch &&\
    chmod -R o+rwx /home/compass_user/scratch

WORKDIR /home/compass_user/OPERA
SHELL ["conda", "run", "-n", "COMPASS", "/bin/bash", "-c"]

RUN python -m pip install ./s1-reader &&\
    cd $HOME/OPERA &&\
    python -m pip install ./COMPASS

WORKDIR /home/compass_user/scratch

ENTRYPOINT ["conda", "run", "-n", "COMPASS","s1_cslc.py"]
