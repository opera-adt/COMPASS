# default docker image for Docker image. User can provide a different base image as a build argument
ARG BASE_IMAGE=oraclelinux:8.8

FROM ${BASE_IMAGE}

# ARG is out of scope here unless re-declared
ARG BASE_IMAGE

RUN echo "Base image is $BASE_IMAGE"

LABEL author="OPERA ADT" \
    description="s1 cslc 0.5.6 point release" \
    version="0.5.6-final"

# Update the base linux image to date
# Create cache and config directory to be used for the dependencies
RUN yum -y update &&\
    yum -y install curl git &&\
    adduser compass_user &&\
    mkdir -p /.config && chmod -R 777 /.config &&\
    mkdir -p /.cache && chmod -R 777 /.cache

RUN mkdir -p /home/compass_user/OPERA

RUN chown -R compass_user:compass_user /home/compass_user/OPERA &&\
    chmod -R 755 /home/compass_user

USER compass_user

ENV CONDA_PREFIX=/home/compass_user/miniforge3

# install Miniconda
WORKDIR /home/compass_user
RUN curl -sSL "https://github.com/conda-forge/miniforge/releases/download/25.3.0-1/Miniforge3-25.3.0-1-Linux-x86_64.sh" -o miniforge.sh &&\
    bash miniforge.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniforge.sh

ENV PATH=${CONDA_PREFIX}/bin:${PATH}
RUN ${CONDA_PREFIX}/bin/conda init bash

# copy COMPASS and set compass_user as owner
COPY --chown=compass_user:compass_user . /home/compass_user/OPERA/COMPASS

# create CONDA environment
RUN conda create --name "COMPASS" --file /home/compass_user/OPERA/COMPASS/docker/specfile.txt \
    && conda clean -afy

SHELL ["conda", "run", "-n", "COMPASS", "/bin/bash", "-c"]

WORKDIR /home/compass_user/OPERA

# installing OPERA s1-reader
RUN curl -sSL https://github.com/opera-adt/s1-reader/archive/refs/tags/v0.2.5.tar.gz -o s1_reader_src.tar.gz &&\
    tar -xvf s1_reader_src.tar.gz &&\
    ln -s s1-reader-0.2.5 s1-reader &&\
    rm s1_reader_src.tar.gz &&\
    python -m pip install ./s1-reader

# installing RAiDER
RUN mkdir RAiDER &&\
    git clone https://github.com/dbekaert/RAiDER.git RAiDER &&\
    cd RAiDER &&\
    git checkout tags/v0.4.2 &&\
    cd .. &&\
    python -m pip install ./RAiDER &&\
    rm -rf RAiDER

# installing OPERA COMPASS. Finalize the installation.
RUN python -m pip install pytest-recording==0.13.3 &&\
    python -m pip install ./COMPASS &&\
    echo "conda activate COMPASS" >> /home/compass_user/.bashrc &&\
    echo "Base image: $BASE_IMAGE" > /home/compass_user/BASE_IMAGE_INFORMATION.txt

WORKDIR /home/compass_user/scratch

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "COMPASS"]
