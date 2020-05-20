# Thermal Face

Fast face detection in thermal images

> TODO: Write summary (referencing [Fever](https://github.com/maxbbraun/fever)).

## Inference

> TODO: Create Python package with wrapper around trained model.

> TODO: Coral setup. https://coral.ai/docs/accelerator/get-started/ (with maximum operating frequency)

> TODO: Add note about performance.

## Training

> TODO: Write intro.

#### 1. Create the dataset

> TODO: Add non-thermal face database (biasing toward thermal in validation and test sets).

> TODO: Explain steps.

```bash
cd training

LOCATION="us-central1"
TDFACE_DIR="tufts-face-database"
TDFACE_BUCKET="gs://$TDFACE_DIR"

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
gsutil iam ch allUsers:objectViewer $TDFACE_BUCKET

curl -O https://raw.githubusercontent.com/maxbbraun/tdface-annotations/master/bounding-boxes.csv
TDFACE_ANNOTATIONS="bounding-boxes.csv"
TDFACE_AUTOML="automl.csv"

python3 -m venv venv
. venv/bin/activate
pip3 install -r requirements.txt

python automl_convert.py --tdface_dir=$TDFACE_DIR --tdface_bucket=$TDFACE_BUCKET --tdface_annotations=$TDFACE_ANNOTATIONS --tdface_automl=$TDFACE_AUTOML

gsutil cp $TDFACE_AUTOML $TDFACE_BUCKET
```

#### 2. Train the model

Using [Cloud AutoML Vision](https://cloud.google.com/vision/automl/object-detection/docs/quickstart-ui):
- Model objective: **Object detection**
- CSV file on Cloud Storage: **`gs://tufts-face-database/automl.csv`**
- Model type: **Edge**
- Optimize for: **Higher accuracy**
- Node budget: **24 node hours**

> TODO: Add steps for model export.
