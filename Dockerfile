##### General instruction: ######
#docker build -f ./Dockerfile -t opera-adt/compass:latest . #To build the docker image
#to run docker container in interactive mode: docker run -it --name COMPASS opera-adt/compass:latest /bin/bash #To run docker container in interactive mode:
#
#
##### For MacOS with Apple silicon: ######
# docker build --platform linux/amd64 -f ./Dockerfile -t opera-adt/compass:latest . #To build the docker image
# docker run -it --platform linux/amd64 --name COMPASS opera-adt/compass:latest /bin/bash #To run docker container in interactive mode:
#
#
##### Mounting options - example ######
# docker run -it --platform linux/amd64 -v $HOME/Desktop/gslc:/home/compass_user/gslc --name COMPASS opera-adt/compass:latest /bin/bash
#
#

FROM ubuntu:latest

SHELL ["/bin/bash","-c"]

LABEL author="OPERA ADT" \
      description="Dockerfile for OPERA COMPASS" \
      version="2022.0527"

RUN apt-get -y update &&\
    apt-get -y install curl git &&\
    adduser --disabled-password compass_user

USER compass_user 

RUN cd $HOME &&\
    curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh &&\
    bash miniconda.sh -b -p ${HOME}/python/miniconda3
ENV PATH="/home/compass_user/python/miniconda3/bin:${PATH}"
ENV PROJ_LIB=/home/compass_user/python/miniconda3/share/proj

RUN conda config --set show_channel_urls True &&\
    conda config --set channel_priority strict &&\
    conda config --add channels conda-forge &&\
    conda update --all &&\
    conda install -c conda-forge isce3 backoff &&\
    echo "Installing OPERA COMPASS" &&\
    mkdir -p $HOME/OPERA/COMPASS &&\
    cd $HOME/OPERA &&\
    git clone https://github.com/opera-adt/COMPASS.git COMPASS &&\
    conda install -c conda-forge --file COMPASS/requirements.txt &&\
    python -m pip install git+https://github.com/opera-adt/s1-reader.git &&\
    python -m pip install ./COMPASS &&\
    echo "CLEAN UP" &&\
    rm $HOME/miniconda.sh

