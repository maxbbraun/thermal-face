from absl import app
from absl import flags
from absl import logging
import csv
from math import ceil
from os import path
from PIL import Image
import random
import re
from tqdm import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'TDFACE', ['TDFACE', 'WIDERFACE', 'FLIR'], 'Whether '
                  'to convert the Tufts Face Database, the WIDER FACE '
                  'database, or the FLIR ADAS Dataset.')
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
flags.DEFINE_integer('max_image_size', 1024, 'The maximum size in pixels an '
                     'image can be without being distorted.')
flags.DEFINE_integer('min_box_size', 20, 'The minimum size in pixels for a '
                     'bounding box not to get discarded.')
flags.DEFINE_float('training_fraction', 0.8, 'The randomly selected '
                   'fraction of total annotations (not images) to be assigned '
                   'to the training set. The sum of train_fraction, '
                   'validation_fraction, and test_fraction may add up to less '
                   'than 1, but not more.')
flags.DEFINE_float('validation_fraction', 0.1, 'The randomly selected '
                   'fraction of total annotations (not images) to be assigned '
                   'to the validation set. The sum of train_fraction, '
                   'validation_fraction, and test_fraction may add up to less '
                   'than 1, but not more.')
flags.DEFINE_float('test_fraction', 0.1, 'The randomly selected fraction of '
                   'total annotations (not images) to be assigned to the '
                   'validation set. The sum of train_fraction, '
                   'validation_fraction, and test_fraction may add up to less '
                   'than 1, but not more.')
flags.DEFINE_string('automl_out', 'automl.csv', 'The output CSV file for '
                    'Cloud AutoML Vision.')

# The regular expression patterns for parsing WIDER FACE annotations.
IMAGE_FILENAME_PATTERN = re.compile(r'^(\d+--.+\.jpg)$')
NUM_FACES_PATTERN = re.compile(r'^(\d+)$')
FACE_PATTERN = re.compile(r'^(\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) (\d+) '
                          r'(\d+) (\d+) $')

# The CSV pattern for the AutoML output file.
AUTOML_PATTERN = '%s,%s,face,%.4f,%.4f,,,%.4f,%.4f,,\n'

# The directory containing the normalized FLIR ADAS images.
FLIR_NORMALIZED_DIR = 'thermal_normalized'


def random_split(dataset):
    dataset = set(dataset)

    # Translate the fractions into counts and assign rounding remainders to the
    # training set if the fractions add up to one.
    training_count = int(FLAGS.training_fraction * len(dataset))
    validation_count = int(FLAGS.validation_fraction * len(dataset))
    test_count = int(FLAGS.test_fraction * len(dataset))
    if (FLAGS.training_fraction + FLAGS.validation_fraction + FLAGS.test_fraction) == 1:
        training_count += len(dataset) - training_count - validation_count - test_count

    # Randomly sample from the dataset and then from the remaining set.
    training_set = set(random.sample(dataset, training_count))
    dataset -= training_set
    validation_set = set(random.sample(dataset, validation_count))
    dataset -= validation_set
    test_set = set(random.sample(dataset, test_count))

    return training_set, validation_set, test_set


def split_label(data, training_set, validation_set, test_set):
    if data in training_set:
        return 'TRAIN'
    if data in validation_set:
        return 'VALIDATE'
    elif data in test_set:
        return 'TEST'
    else:
        return None


def convert_bounding_box(left_str, top_str, width_str, height_str, image_width,
                         image_height):
    left = int(left_str)
    top = int(top_str)
    right = left + int(width_str)
    bottom = top + int(height_str)

    # Clip to the image size.
    left, right = map(lambda x: max(0, min(image_width - 1, x)), [left, right])
    top, bottom = map(lambda y: max(0, min(image_height - 1, y)),
                      [top, bottom])

    # Discard small boxes. Their sizes are calculated after enforcing the
    # maximum image size while maintaining the aspect ratio.
    # https://cloud.google.com/vision/automl/object-detection/docs/prepare
    max_width_scale = FLAGS.max_image_size / image_width
    max_height_scale = FLAGS.max_image_size / image_height
    scale = min(1, min(max_width_scale, max_height_scale))
    scaled_width = int((right - left) * scale)
    scaled_height = int((bottom - top) * scale)
    if min(scaled_width, scaled_height) < FLAGS.min_box_size:
        return None

    # Convert to relative coordinates.
    relative_left = left / image_width
    relative_top = top / image_height
    relative_right = right / image_width
    relative_bottom = bottom / image_height

    return relative_left, relative_top, relative_right, relative_bottom


def main(_):
    dataset = []
    num_rejected = 0

    if FLAGS.mode in ['TDFACE', 'FLIR']:
        # Input formats:
        # https://github.com/maxbbraun/tdface-annotations
        # https://github.com/maxbbraun/flir-adas-faces
        with open(FLAGS.tdface_annotations, newline='') as input_file:
            reader = csv.reader(input_file)
            next(reader)  # Skip header.

            logging.info('Parsing %s annotations.' % FLAGS.mode)
            for row in tqdm(reader):
                if FLAGS.mode == 'TDFACE':
                    local_path = path.join(FLAGS.tdface_dir, *row[:3])
                    gcs_path = path.join(FLAGS.tdface_bucket, *row[:3])
                else:
                    local_path = path.join(FLAGS.tdface_dir, row[0],
                                            FLIR_NORMALIZED_DIR, row[1])
                    gcs_path = path.join(FLAGS.tdface_bucket, row[0],
                                            FLIR_NORMALIZED_DIR, row[1])

                image = Image.open(local_path)
                if FLAGS.mode == 'TDFACE':
                    raw_box = row[3:7]
                else:
                    raw_box = row[2:6]
                bounding_box = convert_bounding_box(*raw_box, image.width,
                                                    image.height)

                if bounding_box:
                    dataset.append((gcs_path, *bounding_box))
                else:
                    num_rejected += 1

    elif FLAGS.mode == 'WIDERFACE':
        # Input format: http://shuoyang1213.me/WIDERFACE/
        with open(FLAGS.widerface_annotations) as input_file:

            logging.info('Parsing WIDERFACE annotations.')
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

                    bounding_box = convert_bounding_box(
                        *face_pattern_match.groups()[:4], image.width,
                        image.height)

                    if bounding_box:
                        dataset.append((gcs_path, *bounding_box))
                    else:
                        num_rejected += 1

                else:
                    raise ValueError('Failed to parse line: %s' % line)

    logging.info('Collected %d annotations and rejected %d.' % (
        len(dataset), num_rejected))

    # Split all available data into the different sets.
    training_set, validation_set, test_set = random_split(dataset)
    logging.info('Writing %s output.' % FLAGS.mode)

    # Output format:
    # https://cloud.google.com/vision/automl/object-detection/docs/csv-format
    with open(FLAGS.automl_out, 'w') as output_file:
        num_skipped = 0
        for data in tqdm(dataset):
            split = split_label(data, training_set, validation_set, test_set)
            if split:
                output_file.write(AUTOML_PATTERN % (split, *data))
            else:
                num_skipped += 1
    logging.info('Wrote %d training, %d validation, and %d test annotations '
                 'and skipped %d.' % (len(training_set), len(validation_set),
                                      len(test_set), num_skipped))


if __name__ == '__main__':
    app.run(main)
