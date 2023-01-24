# Use the nvcr.io/nvidia/pytorch:22.11-py3 base image
FROM nvcr.io/nvidia/pytorch:22.11-py3

# Define the arguments to be passed at build-time
ARG USER_ID
ARG GROUP_ID
ARG USER

# Add a new group and user with the specified GID and UID
RUN addgroup --gid $GROUP_ID $USER \
    && adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER

# Set the MNE_USE_NUMBA environment variable to false
ENV MNE_USE_NUMBA=false

# Copy the current directory to the '/workdir' directory in the container
ADD . /workdir

# Copy the 'docker/meta_requirements.txt' file to the current directory
COPY docker/meta_requirements.txt .

# Install the Python packages listed in the 'meta_requirements.txt' file
RUN pip3 install -r meta_requirements.txt

# Set the working directory to '/workdir'
WORKDIR /workdir
