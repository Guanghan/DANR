FROM nvidia/cuda:10.0-cudnn7-devel
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

RUN pip --no-cache-dir install torch==1.3.1 torchvision==0.4.2 -f https://download.pytorch.org/whl/cu100/torch_stable.html

# install pip packages from requirement
COPY requirements.txt /home/requirements.txt
RUN pip --no-cache-dir install -r /home/requirements.txt

COPY models /home/scanai/models
COPY framework /home/scanai/flag
COPY logs /home/scanai/logs

# set WorkingDir
WORKDIR /home/scanai/flag

#ENTRYPOINT python app.py
ENTRYPOINT bash run.sh
