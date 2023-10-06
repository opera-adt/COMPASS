FROM oraclelinux:8.8

LABEL author="OPERA ADT" \
    description="s1 cslc 0.5.3 point release" \
    version="0.5.3-point"

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

ENV CONDA_PREFIX=/home/compass_user/miniconda3

# install Miniconda
WORKDIR /home/compass_user
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${CONDA_PREFIX} &&\
    rm $HOME/miniconda.sh

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
RUN curl -sSL https://github.com/opera-adt/s1-reader/archive/refs/tags/v0.2.3.tar.gz -o s1_reader_src.tar.gz &&\
    tar -xvf s1_reader_src.tar.gz &&\
    ln -s s1-reader-0.2.3 s1-reader &&\
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


# installing OPERA COMPASS
RUN python -m pip install ./COMPASS &&\
    echo "conda activate COMPASS" >> /home/compass_user/.bashrc

WORKDIR /home/compass_user/scratch

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "COMPASS"]
