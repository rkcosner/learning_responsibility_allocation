# a docker container from https://ngc.nvidia.com/catalog/containers/ with a pytorch installed
FROM nvcr.io/nvidia/pytorch:21.08-py3

# specifying which python version
ARG PYTHON_VERSION=3.8

# installing other nice functionalities
RUN apt-get update
RUN apt-get install htop -y
RUN apt-get install screen -y
RUN apt-get install tmux -y
RUN apt-get install sshfs -y

# installing other packages
RUN pip uninstall tensorboard nvidia-tensorboard -y
RUN pip install tensorboard

# where to cd into at the start
WORKDIR /workspace/