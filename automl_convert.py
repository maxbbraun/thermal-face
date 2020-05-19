from absl import app
from absl import flags
from absl import logging
import csv
from os import path
from PIL import Image

FLAGS = flags.FLAGS
flags.DEFINE_string('tdface_dir', 'tufts-face-database', 'The local directory '
                    'containing the Tufts Face Database image files.')
flags.DEFINE_string('tdface_bucket', 'gs://tufts-face-database', 'The Cloud '
                    'Storage bucket containing the Tufts Face Database image '
                    'files.')
flags.DEFINE_string('tdface_annotations', 'bounding-boxes.csv', 'The input '
                    'CSV file with bounding boxes for the Tufts Face Database '
                    'images.')
flags.DEFINE_string('tdface_automl', 'tdface-automl.csv', 'The output CSV '
                    'file for Cloud AutoML Vision.')


def main(_):
    # Input format: https://github.com/maxbbraun/tdface-annotations
    # Output format:
    # https://cloud.google.com/vision/automl/object-detection/docs/csv-format
    with open(FLAGS.tdface_automl, 'w') as output_file:
        with open(FLAGS.tdface_annotations, newline='') as input_file:
            reader = csv.reader(input_file)
            next(reader)  # Skip header.
            for row in reader:
                local_path = path.join(FLAGS.tdface_dir, *row[:3])
                gcs_path = path.join(FLAGS.tdface_bucket, *row[:3])
                image = Image.open(local_path)
                x_min = int(row[3]) / image.width
                y_min = int(row[4]) / image.height
                x_max = (int(row[3]) + int(row[5])) / image.width
                y_max = (int(row[4]) + int(row[6])) / image.height
                output_file.write(
                    "UNASSIGNED,%s,face,%.4f,%.4f,,,%.4f,%.4f,,\n" % (
                        gcs_path, x_min, y_min, x_max, y_max))


if __name__ == '__main__':
    app.run(main)
