#! /bin/bash

REPO=opera
IMAGE=cslc_s1
TAG=final_0.5.5

echo "IMAGE is $REPO/$IMAGE:$TAG"

rm docker/dockerimg_cslc_s1_${TAG}.tar

# fail on any non-zero exit codes
set -ex

docker build --rm --force-rm --network host -t $REPO/$IMAGE:$TAG -f docker/Dockerfile.isce3_builder .

docker save $REPO/$IMAGE:$TAG > docker/dockerimg_cslc_s1_${TAG}.tar