from absl import app
from absl import flags
from absl import logging
import csv
from os import path
from PIL import Image
import re
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'TDFACE', ['TDFACE', 'WIDERFACE'], 'Whether to '
                  'convert the Tufts Face Database or the WIDER FACE '
                  'database.')
flags.DEFINE_string('tdface_dir', 'tufts-face-database', 'The local directory '
                    'containing the Tufts Face Database image files.')
flags.DEFINE_string('tdface_bucket', 'gs://tufts-face-database', 'The Cloud '
                    'Storage bucket containing the Tufts Face Database image '
                    'files.')
flags.DEFINE_string('tdface_annotations', 'bounding-boxes.csv', 'The input '
                    'CSV file with bounding boxes for the Tufts Face Database '
                    'images.')
flags.DEFINE_string('widerface_dir', 'wider-face', 'The local directory '
                    'containing the WIDER FACE image files.')
flags.DEFINE_string('widerface_bucket', 'gs://wider-face', 'The Cloud Storage '
                    'bucket containing the WIDER FACE image files.')
flags.DEFINE_string('widerface_annotations', 'wider_face_bbx_gt.txt', 'The '
                    'input CSV file with bounding boxes for the WIDER FACE '
                    'database images.')
flags.DEFINE_string('automl_out', 'automl.csv', 'The output CSV file for '
                    'Cloud AutoML Vision.')
# TODO: Add flags for TRAIN/VALIDATE/TEST split fractions.

# The regular expression patterns for parsing WIDER FACE annotations.
IMAGE_FILENAME_PATTERN = re.compile(r'^(\d+--.+\.jpg)$')
NUM_FACES_PATTERN = re.compile(r'^(\d+)$')
FACE_PATTERN = re.compile(r'^(\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) '
                          r'(\d+) (\d+) $')

# The CSV pattern for the AutoML output file.
AUTOML_PATTERN = 'UNASSIGNED,%s,face,%.4f,%.4f,,,%.4f,%.4f,,\n'


def main(_):
    # Output format:
    # https://cloud.google.com/vision/automl/object-detection/docs/csv-format
    with open(FLAGS.automl_out, 'w') as output_file:
        if FLAGS.mode == 'TDFACE':
            # Input format: https://github.com/maxbbraun/tdface-annotations
            with open(FLAGS.tdface_annotations, newline='') as input_file:
                reader = csv.reader(input_file)
                next(reader)  # Skip header.
                for row in tqdm(reader):
                    local_path = path.join(FLAGS.tdface_dir, *row[:3])
                    gcs_path = path.join(FLAGS.tdface_bucket, *row[:3])
                    image = Image.open(local_path)
                    left = int(row[3])
                    top = int(row[4])
                    right = (int(row[3]) + int(row[5]))
                    bottom = (int(row[4]) + int(row[6]))
                    output_file.write(AUTOML_PATTERN % (
                        gcs_path, left / image.width, top / image.height,
                        right / image.width, bottom / image.height))
        elif FLAGS.mode == 'WIDERFACE':
            # Input format: http://shuoyang1213.me/WIDERFACE/
            with open(FLAGS.widerface_annotations) as input_file:
                for count, line in tqdm(enumerate(input_file)):
                    image_filename_match = IMAGE_FILENAME_PATTERN.match(line)
                    num_faces_match = NUM_FACES_PATTERN.match(line)
                    face_pattern_match = FACE_PATTERN.match(line)
                    if image_filename_match:
                        image_filename = image_filename_match.group(1)
                        local_path = path.join(FLAGS.widerface_dir, 'images',
                                               image_filename)
                        image = Image.open(local_path)
                        gcs_path = path.join(FLAGS.widerface_bucket, 'images',
                                             image_filename)
                        image_line_count = count
                    elif num_faces_match:
                        assert count == image_line_count + 1
                        num_faces = int(num_faces_match.group(1))
                    elif face_pattern_match:
                        if not num_faces:
                            # Empty bounding box after 0 face count.
                            continue
                        assert count <= image_line_count + 1 + num_faces
                        if face_pattern_match.group(8) == '1':
                            # Invalid bounding box.
                            continue
                        left = int(face_pattern_match.group(1))
                        top = int(face_pattern_match.group(2))
                        width = int(face_pattern_match.group(3))
                        height = int(face_pattern_match.group(4))
                        right = (left + width)
                        bottom = (top + height)
                        output_file.write(AUTOML_PATTERN % (
                            gcs_path, left / image.width, top / image.height,
                            right / image.width, bottom / image.height))
                    else:
                        raise ValueError('Failed to parse line: %s' % line)


if __name__ == '__main__':
    app.run(main)
