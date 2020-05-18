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
export EXPORT_DIR=${BUCKET}/export
export LOCATION=us-central1
export ZONE=${LOCATION}-a
export VM_NAME=thermal-face-vm
export TF_VERSION=2.2

gsutil mb -l ${LOCATION} -b on ${BUCKET}
# TODO: Copy training data into DATA_DIR.
# https://github.com/maxbbraun/tdface-annotations
# https://www.lfb.rwth-aachen.de/bibtexupload/pdf/KCZ18d.pdf
# https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

ctpu up \
  --vm-only \
  --preemptible-vm \  # TODO
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
export EXPORT_DIR=${BUCKET}/export
export LOCATION=us-central1
export ZONE=${LOCATION}-a
export VM_NAME=thermal-face-vm
export TF_VERSION=2.2

ctpu up \
  --tpu-only \
  --preemptible \  # TODO
  --name=${VM_NAME} \
  --tpu-size=v3-8  \
  --zone=${ZONE} \
  --tf-version=${TF_VERSION}

sudo pip3 install tensorflow-addons tensorflow-model-optimization
export PYTHONPATH="${PYTHONPATH}:/usr/share/models/"

cd /usr/share/tpu/models/official/efficientnet/

nohup tensorboard --logdir=${MODEL_DIR} > /dev/null 2>&1 &

nohup python3 main.py \
  --tpu_name=${VM_NAME} \
  --tpu_zone=${ZONE} \
  --data_dir=${DATA_DIR} \
  --model_dir=${MODEL_DIR} \
  --export_dir=${EXPORT_DIR} \
  --model_name=efficientnet-lite4 \
  --num_label_classes=2 \
  --augment_name=randaugment \
  --train_batch_size=2048 \
  --eval_batch_size=1024 \
  --train_steps=218949 \  # TODO
  --num_train_images=1281167 \  # TODO
  --num_eval_images=50000 \  # TODO
  --steps_per_eval=6255 \  # TODO
  &

ctpu delete --name=${VM_NAME} --zone=${ZONE} --tpu-only
ctpu delete --name=${VM_NAME} --zone=${ZONE}
ctpu status --zone=${ZONE}

# TODO: Download model snapshot.
# TODO: Convert model to TensorFlow Lite
# https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite

gsutil rm -r ${BUCKET}
```
