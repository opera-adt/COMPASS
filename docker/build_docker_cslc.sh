#!/bin/bash


#Example: build_docker_cslc.sh v0.1.2 v0.1.2 opera-adt/cslc:interface_test
ver_s1_reader=NOT_SPECIFIED
ver_compass=NOT_SPECIFIED
str_tag=NOT_SPECIFIED
echo $#
if [ $# -eq 3 ]; then
    ver_s1_reader=$1
    ver_compass=$2
    str_tag=$3
fi


echo 'Version for s1-reader:' $ver_s1_reader
echo '  Version for COMPASS:' $ver_compass
echo '              str_tag:' $str_tag

mkdir -p ./docker/s1-reader
mkdir -p ./docker/COMPASS

cd docker

git clone git@github.com:opera-adt/s1-reader.git s1-reader
cd s1-reader

if [ $ver_s1_reader == NOT_SPECIFIED ]; then
    echo "Using the most recent commit in the repo for s1-reader"
else
    git checkout tags/$ver_s1_reader
fi     

cd ..


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


rm -rf docker/COMPASS
rm -rf docker/s1-reader
