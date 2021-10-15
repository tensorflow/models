from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

import tensorflow as tf

import multiprocessing as mp
from official.vision.beta.data import tfrecord_lib
from research.object_detection.utils import dataset_util
from official.vision.beta.data.tfrecord_lib import convert_to_feature

flags.DEFINE_multi_string('shapenet_dir', '', 'Directory containing ShapeNet.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

def create_tf_example(image):
    with tf.io.gfile.GFile(os.join(image["shapenet_dir"], image["img"]), 'rb') as fid:
        encoded_img_jpg = fid.read()

    img_width, img_height = image["img_size"]
    img_filename = image["img"]
    img_category = image["category"]
    keypoints_2d = image["2d_keypoints"]

    feature_dict = {"img/height": convert_to_feature(img_height),
                    "img/width": convert_to_feature(img_width),
                    "img/category": convert_to_feature(img_category),
                    "img/filename": convert_to_feature(img_filename),
                    "img/encoded": convert_to_feature(encoded_img_jpg),
                    "img/2d_keypoints": convert_to_feature(keypoints_2d)}

    with tf.io.gfile.GFile(os.join(image["shapenet_dir"], image["mask"]), 'rb') as fid:
        encoded_mask_jpg = fid.read()
    
    feature_dict.update({"mask": convert_to_feature(encoded_mask_jpg)})

    with tf.io.gfile.GFile(os.join(image["shapenet_dir"], image["model"]), 'rb') as fid:
        encoded_model = fid.read()

    with tf.io.gfile.GFile(os.join(image["shapenet_dir"], image["3d_keypoints"]), 'rb') as fid:
        keypoints_3d = fid.read()

    model_raw = image["model_raw"]
    model_source = image["model_source"]

    feature_dict.update({"model": convert_to_feature(encoded_model),
                         "model/raw": convert_to_feature(model_raw),
                         "model/source": convert_to_feature(model_source),
                         "model/3d_keypoints": convert_to_feature(keypoints_3d)})

    with tf.io.gfile.GFile(os.join(image["shapenet_dir"], image["voxel"]), 'rb') as fid:
        encoded_voxel = fid.read()

    rot_mat = image["rot_mat"]
    trans_mat = image["trans_mat"]
    focal_length = image["focal_length"]
    cam_position = image["cam_positon"]
    inplane_rotation = image["inplane_rotation"]
    truncated = image["truncated"]
    occluded = image["occluded"]
    slightly_occluded = image["slightly_occluded"]
    bbox = image["bbox"]

    feature_dict.update({"voxel": convert_to_feature(encoded_voxel),
                         "rot_mat": convert_to_feature(rot_mat),
                         "trans_mat": convert_to_feature(trans_mat),
                         "focal_length": convert_to_feature(focal_length),
                         "cam_position": convert_to_feature(cam_position),
                         "inplane_rotation": convert_to_feature(inplane_rotation),
                         "truncated": convert_to_feature(truncated),
                         "occluded": convert_to_feature(occluded),
                         "slightly_occluded": convert_to_feature(slightly_occluded),
                         "bbox": convert_to_feature(bbox)})
    
    example = tf.train.Example(
        features=tf.train.Features(feature=feature_dict))

    return example, 0

def generate_annotations(images, shapenet_dir):
    for image in images:
        yield {"img": image["img"], "category": image["category"], "img_size": image["img_size"], "2d_keypoints": image["2d_keypoints"],
               "mask": image["mask"], "img_source": image["img_source"], "model": image["model"], "model_raw": image["model_raw"],
               "model_source": image["model_source"], "3d_keypoints": image["3d_keypoints"], "voxel": image["voxel"], "rot_mat": image["rot_mat"],
               "trans_mat": image["trans_mat"], "focal_length": image["focal_length"], "cam_position": image["cam_position"],
               "inplane_rotation": image["inplane_rotation"], "truncated": image["truncated"], "occluded": image["occluded"],
               "slightly_ocluded": image["slightly_occluded"], "bbox": image["bbox"], "shapenet_dir": shapenet_dir}

def _create_tf_record_from_shapenet_dir(shapenet_dir,
                                     output_path,
                                     num_shards):
    """Loads Shapenet json files and converts to tf.Record format.
    Args:
      images_info_file: shapenet_dir download directory
      output_path: Path to output tf.Record file.
      num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)

    images = json.load(open(os.join(shapenet_dir, "taxonomy.json")))

    shapenet_annotations_iter = generate_annotations(
        images=images, shapenet_dir=shapenet_dir)

    num_skipped = tfrecord_lib.write_tf_record_dataset(
        output_path, shapenet_annotations_iter, create_tf_example, num_shards)

    logging.info('Finished writing, skipped %d annotations.', num_skipped)

def main(_):
    assert FLAGS.shapenet_dir, '`shapenet_dir` missing.'

    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)


if __name__ == '__main__':
    app.run(main)
