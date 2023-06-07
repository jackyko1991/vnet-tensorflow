# syntax=docker/dockerfile:1
   
FROM tensorflow/tensorflow:1.15.5-gpu-py3
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update && apt-get install -y git
WORKDIR /app
VOLUME ["/app/data","/app/configs","/app/log","/app/ckpt"]
RUN git clone https://github.com/jackyko1991/vnet-tensorflow
# COPY . /app/vnet-tensorflow
WORKDIR /app/vnet-tensorflow
RUN pip install -r requirements.txt
EXPOSE 6006