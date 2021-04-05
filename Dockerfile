FROM nvidia/cuda:9.2-cudnn7-devel
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y python3-opencv ca-certificates python3-dev git wget vim ssh redis-server sudo
RUN apt-get update
RUN apt-get install -y iputils-ping htop

# link python: python = python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# instal pip
RUN wget https://bootstrap.pypa.io/get-pip.py && \
  python get-pip.py && \
  rm get-pip.py

RUN pip --no-cache-dir install torch==1.5.0 torchvision==0.6.0 -f https://download.pytorch.org/whl/cu92/torch_stable.html

# install pip packages from requirement
COPY DANR-det/requirements.txt /home/requirements.txt
RUN pip --no-cache-dir install -r /home/requirements.txt

COPY DANR-det /home/DANR
COPY data /home/DANR/data
COPY weights /home/DANR/weights

# set WorkingDir
WORKDIR /home/DANR

ENTRYPOINT bash
