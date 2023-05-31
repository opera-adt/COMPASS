#! /bin/bash

TAG = opera/cslc_s1:calval_0.3

docker build --rm --force-rm --network host -t $TAG -f docker/Dockerfile .
