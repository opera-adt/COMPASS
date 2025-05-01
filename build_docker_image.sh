#! /bin/bash

REPO=opera
IMAGE=cslc_s1
TAG=final_0.5.6

docker_build_args="--rm --force-rm --network host -t $REPO/$IMAGE:$TAG -f docker/Dockerfile"

if [ $# -eq 0 ]; then
    echo "Base image was not specified. Using the default image specified in the Dockerfile."
else
    echo "Using $1 as the base image."
    docker_build_args+=" --build-arg BASE_IMAGE=$1 "
fi

echo "IMAGE is $REPO/$IMAGE:$TAG"

rm docker/dockerimg_cslc_s1_${TAG}.tar

# fail on any non-zero exit codes
set -ex

#docker build --rm --force-rm --network host -t $REPO/$IMAGE:$TAG -f docker/Dockerfile.isce3_builder .
docker build $docker_build_args .

docker save $REPO/$IMAGE:$TAG > docker/dockerimg_cslc_s1_${TAG}.tar
