FROM nvcr.io/nvidia/pytorch:22.11-py3

ARG USER_ID
ARG GROUP_ID
ARG USER
RUN addgroup --gid $GROUP_ID $USER
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

ENV MNE_USE_NUMBA=false

ADD . workdir

COPY docker/meta_requirements.txt .
RUN pip3 install -r meta_requirements.txt
