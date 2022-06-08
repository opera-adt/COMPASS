#Command to build:
#docker run --rm -it -v [workdir in localhost]:[workdir in the container] opera-adt/compass:latest /home/compass_user/scratch/runconfig_cslc_t64_iw2_b204_20220501.yaml
#
#Command to execute:
# docker run --rm -it -v $HOME/opera-adt/scratch/CSLC_IF_delivery:/home/compass_user/scratch opera-adt/compass:latest --help

FROM ubuntu:22.04

LABEL author="OPERA ADT" \
      description="s1 cslc 0.1.0 IF release" \
      version="0.1.0_IF"

RUN apt-get -y update &&\
    apt-get -y install curl git &&\
    adduser --disabled-password compass_user

USER compass_user 

COPY specifile.txt /home/compass_user/

ENV CONDA_PREFIX=/home/compass_user/miniconda3


WORKDIR /home/compass_user
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX}

ENV PATH="${CONDA_PREFIX}/bin:${PATH}"
RUN $CONDA_PREFIX/bin/conda init bash

RUN conda config --set show_channel_urls True &&\
    conda config --set channel_priority strict &&\
    conda config --add channels conda-forge &&\
    conda update --all &&\
    conda create --name "COMPASS" --file specifile.txt

SHELL ["conda", "run", "-n", "COMPASS", "/bin/bash", "-c"]

RUN echo "Installing OPERA COMPASS" &&\
    mkdir -p $HOME/OPERA &&\
    cd $HOME/OPERA &&\
    curl -sSL https://github.com/opera-adt/COMPASS/archive/refs/tags/v0.1.1.tar.gz -o compass_src.tar.gz &&\
    curl -sSL https://github.com/opera-adt/s1-reader/archive/refs/tags/v0.1.0.tar.gz -o s1_reader_src.tar.gz &&\
    tar -xvf compass_src.tar.gz &&\
    tar -xvf s1_reader_src.tar.gz &&\
    ln -s COMPASS-0.1.1 COMPASS &&\
    ln -s s1-reader-0.1.0 s1-reader &&\    
    conda install -c conda-forge --file COMPASS/requirements.txt &&\
    conda install -c conda-forge --file s1-reader/requirements.txt &&\
    python -m pip install ./COMPASS &&\
    python -m pip install ./s1-reader &&\
    cd $HOME &&\
    rm $HOME/miniconda.sh

WORKDIR /home/compass_user/scratch
#ENTRYPOINT ["conda", "run", "-n", "COMPASS", "python","/home/compass_user/miniconda3/envs/COMPASS/bin/s1_cslc.py"] #for COMPASS 0.1.0
ENTRYPOINT ["conda", "run", "-n", "COMPASS","s1_cslc.py"] #for COMPASS 0.1.1
