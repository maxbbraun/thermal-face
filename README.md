# Thermal Face

**Thermal Face** is a machine learning model for fast face detection in thermal images. It was built for [**Fever**](https://github.com/maxbbraun/fever), the contactless fever thermometer with auto-aim.

## Inference

The face detection model is using [TensorFlow Lite](https://www.tensorflow.org/lite) for optimal performance on mobile/edge devices. The recommended inference setup is a [Raspberry Pi 4 Model B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) with a [Coral USB Accelerator](https://coral.ai/docs/accelerator/get-started/).

The following is an example for inference from Python on an image file using the compiled model [`thermal_face_automl_edge_l_edgetpu.tflite`](models/thermal_face_automl_edge_l_edgetpu.tflite) and the [Edge TPU API](https://coral.ai/docs/edgetpu/api-intro/):

```bash
pip3 install Pillow
sudo apt-get install python3-edgetpu
```

```python
from edgetpu.detection.engine import DetectionEngine
from PIL import Image

# One-time initialization:
face_detector = DetectionEngine('thermal_face_automl_edge_l_edgetpu.tflite')

# Per-image detection:
image = Image.open('image.png').convert('RGB')
faces = face_detector.detect_with_image(image,
    threshold=0.5
    top_k=10,
    keep_aspect_ratio=True,
    relative_coord=False,
    resample=Image.BILINEAR)
for face in faces:
  # np.array([[left, top], [right, bottom]], dtype=float64)
  face.bounding_box
```

You can also use the [TF Lite API](https://www.tensorflow.org/lite/guide/python) directly on the compiled model or, in the absence of a Edge TPU, on the uncompiled model [`thermal_face_automl_edge_l.tflite`](models/thermal_face_automl_edge_l.tflite).

> TODO: Add a note about expected performance numbers. Mention Coral maximum operating frequency. Profile inference alone vs. image conversion etc. Try Raspberry Pi overclocking.

## Training

> TODO: Explain strategy for dataset and training.

#### 1. Create the dataset

> TODO: Add non-thermal face database (biasing toward thermal in validation and test sets): [WIDER_train.zip](https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDQUUwd21EckhUbWs), [WIDER_val.zip](https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDd3dIRmpvSk8tLUk), [WIDER_test.zip](https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDbW4tdGpaYjgzZkU), [wider_face_split.zip](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip)

Download the thermal images from the [Tufts Face Database](http://tdface.ece.tufts.edu) and upload them to [Cloud Storage](https://cloud.google.com/storage/docs):

```bash
cd training

LOCATION="us-central1"
TDFACE_DIR="tufts-face-database"
TDFACE_BUCKET="gs://$TDFACE_DIR"
MODEL_BUCKET="gs://thermal-face"

for i in $(seq 1 4)
do
  curl -O http://tdface.ece.tufts.edu/downloads/TD_IR_A/TD_IR_A_Set$i.zip
  curl -O http://tdface.ece.tufts.edu/downloads/TD_IR_E/TD_IR_E_Set$i.zip
done

mkdir $TDFACE_DIR
for f in TD_IR_*.zip
do
  unzip $f -d $TDFACE_DIR/$(basename $f .zip)
done

gsutil mb -l $LOCATION $TDFACE_BUCKET
gsutil -m rsync -r $TDFACE_DIR $TDFACE_BUCKET
gsutil mb -l $LOCATION $MODEL_BUCKET
```

Create a dataset spec in the [AutoML format](https://cloud.google.com/vision/automl/object-detection/docs/csv-format) using the matching [bounding box annotations](https://github.com/maxbbraun/tdface-annotations):

```bash
curl -O https://raw.githubusercontent.com/maxbbraun/tdface-annotations/master/bounding-boxes.csv
TDFACE_ANNOTATIONS="bounding-boxes.csv"
AUTOML_SPEC="automl.csv"

python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt

python automl_convert.py \
  --tdface_dir=$TDFACE_DIR \
  --tdface_bucket=$TDFACE_BUCKET \
  --tdface_annotations=$TDFACE_ANNOTATIONS \
  --tdface_automl=$AUTOML_SPEC

gsutil cp $AUTOML_SPEC $MODEL_BUCKET
```

#### 2. Train the model

```bash
MODEL_NAME="thermal_face_automl_edge_l"
```

Using [Cloud AutoML Vision](https://cloud.google.com/vision/automl/object-detection/docs/edge-quickstart):

> TODO: Try optimizing for best trade-off instead.

 - Model objective: **Object detection**
 - CSV file on Cloud Storage: **`$MODEL_BUCKET/$AUTOML_SPEC`**
 - Model name: **`$MODEL_NAME`**
 - Model type: **Edge**
 - Optimize for: **Higher accuracy**
 - Node budget: **24 node hours**
 - Use model: **TF Lite**
 - Export to Cloud Storage: **`$MODEL_BUCKET/`**

#### 3. Compile the model

> TODO: Fix issues with TPU-compiled model.

Use [Docker](https://docs.docker.com) to compile the model for [Edge TPU](https://coral.ai/products/):

```bash
MODEL_FILE="$MODEL_NAME.tflite"
TPU_MODEL_FILE="${MODEL_FILE%.*}_edgetpu.${MODEL_FILE##*.}"
COMPILER_NAME="edgetpu_compiler"

gsutil cp $MODEL_BUCKET/**/*$MODEL_NAME*/model.tflite $MODEL_FILE

docker build \
  --file $COMPILER_NAME.Dockerfile \
  --tag $COMPILER_NAME \
  --build-arg MODEL_FILE=$MODEL_FILE \
  .
docker run $COMPILER_NAME
docker cp $(docker ps -alq):/$TPU_MODEL_FILE .

mv $MODEL_FILE ../models/
mv $TPU_MODEL_FILE ../models/
```
