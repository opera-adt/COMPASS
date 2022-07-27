#!/bin/bash

#ver_s1_reader=v0.1.2
#ver_compass=v0.1.2

if [ "$#" -eq 3 ]; then
    ver_s1_reader=$1
    ver_compass=$2
fi


mkdir s1-reader
mkdir COMPASS

git clone git@github.com:opera-adt/s1-reader.git s1-reader
cd s1-reader
git checkout tags/$ver_s1_reader

cd ..


git clone git@github.com:opera-adt/COMPASS.git COMPASS
cd COMPASS
git checkout tags/$ver_compass

cd ..
docker build -f docker/Dockerfile -t opera-adt/cslc:interface_test .

rm -rf COMPASS
rm -rf s1-reader
