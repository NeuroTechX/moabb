#!/bin/bash
# A simple script to build the distributed Docker image.
#
# $ create_docker_image.sh
set -ex
# Updating the repository or (cloning the repository and Navigate into the repository)
git -C $TAG pull || true

# Build the Docker image
docker build . -f Dockerfile -t "${TAG}"  \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg USER=${USER}
