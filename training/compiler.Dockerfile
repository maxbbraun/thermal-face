FROM tensorflow/tensorflow:1.15.2-py3

RUN apt-get update
RUN apt-get -y install curl

# https://coral.ai/docs/edgetpu/compiler/
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update
RUN apt-get -y install edgetpu-compiler

# Turn the build arguments into environment variables.
ARG MODEL_FILE
ENV MODEL_FILE $MODEL_FILE
ARG OUT_DIR
ENV OUT_DIR $OUT_DIR

# Add the model file to be compiled.
ADD $MODEL_FILE .

ENTRYPOINT [ "sh", "-c", "edgetpu_compiler --show_operations --out_dir=$OUT_DIR $MODEL_FILE" ]
