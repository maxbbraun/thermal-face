# ThermalFace

Face detection in thermal images

## Training

- [EfficientNets](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
- [Training EfficientNet on Cloud TPU](https://cloud.google.com/tpu/docs/tutorials/efficientnet-2.x)

```bash
export BUCKET_NAME=thermal-face-training
export BUCKET=gs://${BUCKET_NAME}
export MODEL_DIR=${BUCKET}/model
export DATA_DIR=${BUCKET}/data
export LOCATION=us-central1
export ZONE=${LOCATION}-a
export VM_NAME=thermal-face-vm
export TF_VERSION=2.2

gsutil mb -l ${LOCATION} -b on ${BUCKET}
# TODO: Copy training data into DATA_DIR.
# https://github.com/maxbbraun/tdface-annotations
# https://www.lfb.rwth-aachen.de/bibtexupload/pdf/KCZ18d.pdf

ctpu up \
  --vm-only \
  --name=${VM_NAME} \
  --zone=${ZONE} \
  --disk-size-gb=300 \
  --machine-type=n1-standard-16 \
  --tf-version=${TF_VERSION}

gcloud compute ssh ${VM_NAME} --zone=${ZONE} -- -NfL 6006:localhost:6006
gcloud compute ssh ${VM_NAME} --zone=${ZONE}
```

On VM:

```bash
export BUCKET_NAME=thermal-face-training
export BUCKET=gs://${BUCKET_NAME}
export MODEL_DIR=${BUCKET}/model
export DATA_DIR=${BUCKET}/data
export LOCATION=us-central1
export ZONE=${LOCATION}-a
export VM_NAME=thermal-face-vm
export TF_VERSION=2.2

ctpu up \
  --tpu-only \
  --name=${VM_NAME} \
  --tpu-size=v3-8  \
  --zone=${ZONE} \
  --tf-version=${TF_VERSION}

sudo pip3 install tensorflow-addons tensorflow-model-optimization
export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"

cd /usr/share/models/official/vision/image_classification/

nohup tensorboard --logdir=${MODEL_DIR} > /dev/null 2>&1 &

# TODO: Start with pretrained model snapshot.

nohup python3 classifier_trainer.py \
  --mode=train_and_eval \
  --model_type=efficientnet \
  --dataset=tdface \
  --tpu=${VM_NAME} \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  # TODO: Replace with custom config.
  --config_file=configs/examples/efficientnet/imagenet/efficientnet-b0-tpu.yaml &

ctpu delete --name=${VM_NAME} --zone=${ZONE} --tpu-only
ctpu delete --name=${VM_NAME} --zone=${ZONE}
ctpu status --zone=${ZONE}

# TODO: Download model snapshot.
# TODO: Convert to TensorFlow Lite: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite

gsutil rm -r ${BUCKET}
```
