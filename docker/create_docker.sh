#!/bin/bash
# A simple script to build the distributed Docker image.
#
# $ create_docker_image.sh
set -ex
TAG=moabb

docker build . -f Dockerfile -t "${TAG}"  \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}
