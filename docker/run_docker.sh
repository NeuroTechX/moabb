#!/bin/bash
set -ex # Enable 'set -e' (exit on error) and 'set -x' (debugging) options

TAG=moabb
MOUNT_POINT="/home/bru/PycharmProjects/moabb/dataset/:/root/mne_data/"
PIPELINE="/workdir/pipelines/"
RESULTS="/workdir/results"
OUTPUTS="/workdir/outputs/"
DATASET="/workdir/dataset"

# Run a Docker container with the following options:
# -it : interactive and tty mode
# -v : mount a volume from host machine at $MOUNT_POINT to container's /root/mne_data/
# "${TAG}" : use the image with the tag 'moabb'
# /usr/bin/python : use python command
# /workdir/moabb/run.py : run the script 'run.py' located at '/workdir/moabb/'
# --pipeline $PIPELINE : use pipelines located at $PIPELINE
# --results $RESULTS : store results in $RESULTS
# --output $OUTPUTS : store outputs in $OUTPUTS
# --mne_data $DATASET : use dataset located at $DATASET
docker run -it -v "$MOUNT_POINT" \
    "$TAG" \
    /usr/bin/python \
    /workdir/moabb/run.py \
    --pipeline "$PIPELINE" \
    --results "$RESULTS" \
    --output "$OUTPUTS" \
    --mne_data "$DATASET"
