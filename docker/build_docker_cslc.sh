#!/bin/bash

# A basc script to build the Docker image for COMPASS
#
# Usage:
# build_docker_cslc.sh [s1-reader version] [COMPASS version] [Docker image tag]
# 
# [s1-reader version] : optional
#     version of s1-reader to use in Docker image
# [COMPASS version]: optional
#     Version of COMPASS to use in Docker image
# [Docker image tag]: optional
#     A tag of the output Docker image. See the example command below for example.
#
# Example commands:
#
# build_docker_cslc.sh # run the script with no arguments to 
# build_docker_cslc.sh v0.1.2 v0.1.2 opera-adt/cslc:interface_test #Specify the versions of s1-reader and COMPASS for Docker image to be built
#

ver_s1_reader=NOT_SPECIFIED
ver_compass=NOT_SPECIFIED
str_tag=NOT_SPECIFIED

if [ $# -eq 3 ]; then
    ver_s1_reader=$1
    ver_compass=$2
    str_tag=$3
fi

echo ' '
echo 'Version for s1-reader:' $ver_s1_reader
echo '  Version for COMPASS:' $ver_compass
echo '              str_tag:' $str_tag

mkdir -p ./docker/s1-reader
mkdir -p ./docker/COMPASS

echo ' '
cd docker

git clone git@github.com:opera-adt/s1-reader.git s1-reader
cd s1-reader

if [ $ver_s1_reader == NOT_SPECIFIED ]; then
    echo "Using the most recent commit in the repo for s1-reader"
else
    git checkout tags/$ver_s1_reader
fi     

cd ..

echo ' '
git clone git@github.com:opera-adt/COMPASS.git COMPASS
cd COMPASS

if [ $ver_compass == NOT_SPECIFIED ]; then
    echo "Using the most recent commit in the repo for COMPASS"
else
    git checkout tags/$ver_compass
fi     

cd ../..


if [ $str_tag == NOT_SPECIFIED ]; then
    docker build . -f docker/Dockerfile
else
    docker build . -f docker/Dockerfile -t $str_tag
fi     

echo "Cleaning up"
rm -rf docker/COMPASS
rm -rf docker/s1-reader
