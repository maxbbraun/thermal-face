# Thermal Face

**Thermal Face** is a machine learning model for fast face detection in thermal images. It was built for [**Fever**](https://github.com/maxbbraun/fever), the contactless fever thermometer with auto-aim.

## Inference

The face detection model is using [TensorFlow Lite](https://www.tensorflow.org/lite) for optimal performance on mobile/edge devices. The recommended inference setup is a [Raspberry Pi 4 Model B](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) with a [Coral USB Accelerator](https://coral.ai/docs/accelerator/get-started/).

The following is an example for inference from Python on an image file using the compiled model [`thermal_face_automl_edge_fast_edgetpu.tflite`](models/thermal_face_automl_edge_fast_edgetpu.tflite) and the [Edge TPU API](https://coral.ai/docs/edgetpu/api-intro/):

```bash
pip3 install Pillow
sudo apt-get install python3-edgetpu
```

```python
from edgetpu.detection.engine import DetectionEngine
from PIL import Image

# One-time initialization:
face_detector = DetectionEngine('thermal_face_automl_edge_fast_edgetpu.tflite')

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

You can also use the [TF Lite API](https://www.tensorflow.org/lite/guide/python) directly on the compiled model or, in the absence of a Edge TPU, on the uncompiled model [`thermal_face_automl_edge_fast.tflite`](models/thermal_face_automl_edge_fast.tflite).

## Training

The model is trained with [Cloud AutoML](https://cloud.google.com/automl) using a face dataset that combines a large set of images in visible light from the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) database and a smaller set of thermal images from the [Tufts Face Database](http://tdface.ece.tufts.edu) and the [FLIR ADAS Dataset](https://www.flir.com/oem/adas/adas-dataset-form/).

### 1. Create the dataset

There are a total of 8,852 images and 59,288 face bounding box annotations in the combined dataset. The WIDER FACE set is large and diverse, but only contains visible-light images. The thermal images from the Tufts Face Database and FLIR ADAS Dataset are fewer and less diverse, so we mix the three sets before splitting them into [training, validation, and test sets](https://cloud.google.com/vision/automl/object-detection/docs/prepare). The relative size of the test and validation sets are unusually small to achieve a better balance among the source datasets while still using a significant fraction of all available training data. The exact breakdown of images (and annotations) is as follows:

| | Training set | Validation set | Test set |
| -: | -: | -: | -: |
| **Tufts Face Database (IR)** | 1,247 (1,247) | 155 (155) | 155 (155) |
| _Fraction of source images_ | 80% | 10% | 10% |
| _Fraction of combined_ | 15% (2%) | 42% (10%) | 42% (12%) |
| **FLIR ADAS (Faces)** | 440 (580) | 54 (74) | 54 (74) |
| _Fraction of source images_ | 80% | 10% | 10% |
| _Fraction of combined_ | 5% (1%) | 15% (5%) | 15% (6%) |
| **WIDER FACE (Validation)** | 0 (0) | 160 (1,350) | 160 (1,022) |
| _Fraction of source images_ | 0% | 5% | 5% |
| _Fraction of combined_ | 0% (0%) | 43% (85%) | 43% (82%) |
| **WIDER FACE (Training)** | 6,427 (54,631) | 0 (0) | 0 (0) |
| _Fraction of source images_ | 50% | 0% | 0% |
| _Fraction of combined_ | 79% (97%) | 0% (0%) | 0% (0%) |
| **_Combined_** | 8,114 (56,458) | 369 (1,579) | 369 (1,251) |
| _Fraction of combined sources_ | 92% (95%) | 4% (3%) | 4% (3%) |

#### 1.1 Get the Tufts Face Database

Download the thermal images from the [Tufts Face Database](http://tdface.ece.tufts.edu) and upload them to [Cloud Storage](https://cloud.google.com/storage/docs):

```bash
LOCATION="us-central1"
TDFACE_DIR="tufts-face-database"
TDFACE_BUCKET="gs://$TDFACE_DIR"

cd training

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
```

Create a dataset spec of the images in the [AutoML format](https://cloud.google.com/vision/automl/object-detection/docs/csv-format) using the [separately created](https://github.com/maxbbraun/tdface-annotations) bounding box annotations and upload it:

```bash
TDFACE_ANNOTATIONS="$TDFACE_DIR/bounding-boxes.csv"
TDFACE_AUTOML="tdface-automl.csv"
TDFACE_TRAINING_FRACTION=0.8
TDFACE_VALIDATION_FRACTION=0.1
TDFACE_TEST_FRACTION=0.1

curl https://raw.githubusercontent.com/maxbbraun/tdface-annotations/master/bounding-boxes.csv -o $TDFACE_ANNOTATIONS

python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt

python automl_convert.py \
  --mode=TDFACE \
  --tdface_dir=$TDFACE_DIR \
  --tdface_bucket=$TDFACE_BUCKET \
  --tdface_annotations=$TDFACE_ANNOTATIONS \
  --training_fraction=$TDFACE_TRAINING_FRACTION \
  --validation_fraction=$TDFACE_VALIDATION_FRACTION \
  --test_fraction=$TDFACE_TEST_FRACTION \
  --automl_out=$TDFACE_AUTOML

gsutil cp $TDFACE_AUTOML $TDFACE_BUCKET
```

#### 1.2 Get FLIR ADAS

[Download](https://www.flir.com/oem/adas/adas-dataset-form/) the `FLIR_ADAS_1_3.tar.*` files and process them:

```bash
FLIR_ADAS_REPO="flir-adas-faces"
FLIR_ADAS_DIR="flir-adas-database"
TMP_FLIR_ADAS_DIR="/tmp/$FLIR_ADAS_DIR"
FLIR_ADAS_BUCKET="gs://$FLIR_ADAS_DIR"
FLIR_ADAS_ANNOTATIONS="$FLIR_ADAS_REPO/bounding-boxes.csv"
FLIR_ADAS_AUTOML="flir-adas-automl.csv"
FLIR_ADAS_TRAINING_FRACTION=0.8
FLIR_ADAS_VALIDATION_FRACTION=0.1
FLIR_ADAS_TEST_FRACTION=0.1

git clone https://github.com/maxbbraun/flir-adas-faces.git $FLIR_ADAS_REPO
cd $FLIR_ADAS_REPO

mkdir $FLIR_ADAS_DIR
tar -C $FLIR_ADAS_DIR -xvf FLIR_ADAS_1_3.tar.001 --strip-components=1

python3 -m venv venv
. venv/bin/activate
pip install -r requirements.txt

python flir_convert.py \
  --input_dir=$FLIR_ADAS_DIR/train \
  --output_dir=$TMP_FLIR_ADAS_DIR/train \
  --output_csv=/dev/null
python flir_convert.py \
  --input_dir=$FLIR_ADAS_DIR/val \
  --output_dir=$TMP_FLIR_ADAS_DIR/val \
  --output_csv=/dev/null
python flir_convert.py \
  --input_dir=$FLIR_ADAS_DIR/video \
  --output_dir=$TMP_FLIR_ADAS_DIR/video \
  --output_csv=/dev/null

cd ..
. venv/bin/activate

python automl_convert.py \
  --mode=FLIR \
  --tdface_dir=$TMP_FLIR_ADAS_DIR \
  --tdface_bucket=$FLIR_ADAS_BUCKET \
  --tdface_annotations=$FLIR_ADAS_ANNOTATIONS \
  --training_fraction=$FLIR_ADAS_TRAINING_FRACTION \
  --validation_fraction=$FLIR_ADAS_VALIDATION_FRACTION \
  --test_fraction=$FLIR_ADAS_TEST_FRACTION \
  --automl_out=$FLIR_ADAS_AUTOML

gsutil cp $FLIR_ADAS_AUTOML $FLIR_ADAS_BUCKET
```

#### 1.3 Get WIDER FACE

Download and upload the [WIDER FACE](http://shuoyang1213.me/WIDERFACE/) dataset:
 - [WIDER_train.zip](https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDQUUwd21EckhUbWs)
 - [WIDER_val.zip](https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDd3dIRmpvSk8tLUk)
 - [WIDER_test.zip](https://drive.google.com/uc?export=download&id=0B6eKvaijfFUDbW4tdGpaYjgzZkU)
 - [wider_face_split.zip](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip)

```bash
WIDERFACE_DIR="wider-face-database"
WIDERFACE_BUCKET="gs://$WIDERFACE_DIR"

mkdir $WIDERFACE_DIR
for f in WIDER_*.zip wider_*.zip
do
  unzip $f -d $WIDERFACE_DIR/
done

gsutil mb -l $LOCATION $WIDERFACE_BUCKET
gsutil -m rsync -r $WIDERFACE_DIR $WIDERFACE_BUCKET
```

 Create and upload the AutoML spec using the included bounding boxes:

```bash
WIDERFACE_TRAINING_AUTOML="widerface-training-automl.csv"
WIDERFACE_VALIDATION_AUTOML="widerface-validation-automl.csv"
WIDERFACE_TRAINING_TRAINING_FRACTION=0.5
WIDERFACE_TRAINING_VALIDATION_FRACTION=0
WIDERFACE_TRAINING_TEST_FRACTION=0
WIDERFACE_VALIDATION_TRAINING_FRACTION=0
WIDERFACE_VALIDATION_VALIDATION_FRACTION=0.05
WIDERFACE_VALIDATION_TEST_FRACTION=0.05

python automl_convert.py \
  --mode=WIDERFACE \
  --widerface_dir=$WIDERFACE_DIR/WIDER_train \
  --widerface_bucket=$WIDERFACE_BUCKET/WIDER_train \
  --widerface_annotations=$WIDERFACE_DIR/wider_face_split/wider_face_train_bbx_gt.txt \
  --training_fraction=$WIDERFACE_TRAINING_TRAINING_FRACTION \
  --validation_fraction=$WIDERFACE_TRAINING_VALIDATION_FRACTION \
  --test_fraction=$WIDERFACE_TRAINING_TEST_FRACTION \
  --automl_out=$WIDERFACE_TRAINING_AUTOML
python automl_convert.py \
  --mode=WIDERFACE \
  --widerface_dir=$WIDERFACE_DIR/WIDER_val \
  --widerface_bucket=$WIDERFACE_BUCKET/WIDER_val \
  --widerface_annotations=$WIDERFACE_DIR/wider_face_split/wider_face_val_bbx_gt.txt \
  --training_fraction=$WIDERFACE_VALIDATION_TRAINING_FRACTION \
  --validation_fraction=$WIDERFACE_VALIDATION_VALIDATION_FRACTION \
  --test_fraction=$WIDERFACE_VALIDATION_TEST_FRACTION \
  --automl_out=$WIDERFACE_VALIDATION_AUTOML

gsutil cp $WIDERFACE_TRAINING_AUTOML $WIDERFACE_BUCKET
gsutil cp $WIDERFACE_VALIDATION_AUTOML $WIDERFACE_BUCKET
```

#### 1.4 Combine the datasets

Combine all AutoML dataset specs into one and upload it:

```bash
THERMAL_FACE_AUTOML="automl.csv"
MODEL_BUCKET="gs://thermal-face"
MODEL_NAME="thermal_face_automl_edge_fast"

gsutil mb -l $LOCATION $MODEL_BUCKET

rm -f $THERMAL_FACE_AUTOML
cat $TDFACE_AUTOML >> $THERMAL_FACE_AUTOML
cat $WIDERFACE_TRAINING_AUTOML >> $THERMAL_FACE_AUTOML
cat $WIDERFACE_VALIDATION_AUTOML >> $THERMAL_FACE_AUTOML
cat $FLIR_ADAS_AUTOML >> $THERMAL_FACE_AUTOML

gsutil cp $THERMAL_FACE_AUTOML $MODEL_BUCKET
```

### 2. Train the model

Use [Cloud AutoML Vision](https://cloud.google.com/vision/automl/object-detection/docs/edge-quickstart) with the following options:

- New dataset name: **`tdface_x_widerface_x_fliradas`**
- Model objective: **Object detection**
- CSV file on Cloud Storage: **`$MODEL_BUCKET/$THERMAL_FACE_AUTOML`**
- Model name: **`$MODEL_NAME`**
- Model type: **Edge**
- Optimize for: **Faster predictions**
- Node budget: **24 node hours**
- Use model: **TF Lite**
- Export to Cloud Storage: **`$MODEL_BUCKET/`**

### 3. Compile the model

⚠️ ***See [open issue](https://github.com/maxbbraun/thermal-face/issues/2).***

Use [Docker](https://docs.docker.com) to compile the model for [Edge TPU](https://coral.ai/products/):

```bash
MODEL_FILE="$MODEL_NAME.tflite"
TPU_MODEL_FILE="${MODEL_FILE%.*}_edgetpu.${MODEL_FILE##*.}"
MODEL_DIR="$(pwd)/../models"
COMPILER_NAME="compiler"
OUT_DIR="/out"

gsutil cp $MODEL_BUCKET/**/*$MODEL_NAME*/model.tflite $MODEL_FILE

docker build . \
  --file=$COMPILER_NAME.Dockerfile \
  --tag=$COMPILER_NAME \
  --build-arg=MODEL_FILE=$MODEL_FILE \
  --build-arg=OUT_DIR=$OUT_DIR
docker run \
  --mount=type=bind,source=$MODEL_DIR,target=$OUT_DIR \
  --rm \
  $COMPILER_NAME
mv $MODEL_FILE $MODEL_DIR/
```
