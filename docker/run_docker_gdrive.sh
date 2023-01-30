#!/bin/bash
set -ex # Enable 'set -e' (exit on error) and 'set -x' (debugging) options

TAG=moabb
# Define the repository to download
REPO_URL=https://github.com/NeuroTechX/moabb.git
# Updating the repository or (cloning the repository and Navigate into the repository)
git -C $TAG pull || (git clone $REPO_URL moabb && cd moabb)

MOUNT_POINT=${1} # The first argument is the mount point

# Where to mount the dataset inside the docker container

DATASET="/workdir/dataset/"
# Run a Docker container with the following options:
# -it : interactive and tty mode
# -v : mount a volume from host machine at $MOUNT_POINT to container's /root/mne_data/
# "${TAG}" : use the image with the tag 'moabb'
# --mne_data $DATASET : use dataset located at $DATASET
docker run -it \
    -v "${MOUNT_POINT}/dataset/:/workdir/dataset" \
    "$TAG" \
    bash /workdir/moabb/docker/run_gdown.py \
