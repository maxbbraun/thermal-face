FROM tensorflow/tensorflow:1.15.2-py3

RUN apt-get update
RUN apt-get -y install curl

# https://coral.ai/docs/edgetpu/compiler/
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get -y install edgetpu-compiler

ARG MODEL_FILE
ENV MODEL_FILE $MODEL_FILE
ADD $MODEL_FILE .

ENTRYPOINT [ "sh", "-c", "edgetpu_compiler --show_operations $MODEL_FILE" ]
