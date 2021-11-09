# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# https://github.com/PurdueDualityLab/tf-models/blob/master/official/vision/beta/data/create_coco_tf_record.py reference

r"""Convert raw Pix3D dataset to TFRecord format.
Example usage:
    python create_pix3d_tf_record.py --logtostderr \
      --pix3d_dir="${TRAIN_IMAGE_DIR}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=32 \ 
      --pix3d_json_file="pix3d.json"
"""

import json
import logging
import os
import json

from absl import app  # pylint:disable=unused-import
from absl import flags
import numpy as np

import tensorflow as tf

from official.vision.beta.data.tfrecord_lib import write_tf_record_dataset

def convert_to_feature(value, value_type=None):
    """Converts the given python object to a tf.train.Feature.

    Args:
    value: int, float, bytes or a list of them.
    value_type: optional, if specified, forces the feature to be of the given
        type. Otherwise, type is inferred automatically.
    Returns:
      feature: A tf.train.Feature object.
    """

    if value_type is None:

        element = value[0] if isinstance(value, list) else value

        if isinstance(element, bytes):
            value_type = 'bytes'

        elif isinstance(element, (int, np.integer)):
            value_type = 'int64'

        elif isinstance(element, (float, np.floating)):
            value_type = 'float'

        elif isinstance(element, str):
            value_type = "str"

        elif isinstance(element, list) and isinstance(value, list):
            value_type = "2d"

        elif element is None:
            value_type = "none"

        else:
            raise ValueError('Cannot convert type {} to feature'.
                             format(type(element)))

        if isinstance(value, list):
            value_type = value_type + '_list'

    if value_type == 'int64':
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    elif value_type == 'int64_list':
        value = np.asarray(value).astype(np.int64).reshape(-1)
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    elif value_type == 'float':
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    elif value_type == 'float_list':
        value = np.asarray(value).astype(np.float32).reshape(-1)
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    elif value_type == 'bytes':
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    elif value_type == 'bytes_list':
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    elif value_type == "str":
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value.encode("utf-8")]))

    elif value_type == "2d_list":
        data = tf.convert_to_tensor(value)
        serialized_data = tf.io.serialize_tensor(data)

        return tf.train.Feature(
              bytes_list=tf.train.BytesList(value=[serialized_data.numpy()]))

    elif value_type == "none":
        return tf.train.Feature()

    else:
        raise ValueError(
            'Unknown value_type parameter - {}'.format(value_type))

flags.DEFINE_multi_string('pix3d_dir', '', 'Directory containing Pix3d.')
flags.DEFINE_string('output_file_prefix', '/tmp/train', 'Path to output file')
flags.DEFINE_integer('num_shards', 32, 'Number of shards for output file.')
flags.DEFINE_multi_string("pix3d_json_file", "pix3d.json",
                          "Json file containing all pix3d info")

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def create_tf_example(image):
    """Converts image and annotations to a tf.Example proto.
    Args:
      image: dict with keys: 
        [img, category, img_size, 2d_keypoints, mask, img_source, model,
          model_raw, model_source, 3d_keypoints, voxel, rot_mat, trans_mat,
          focal_length, cam_position, inplane_rotation, truncated, occluded,
          slightly_occluded, bbox]
    Returns:
      example: The converted tf.Example
      num_annotations_skipped: Number of (invalid) annotations that were 
        ignored.
    Raises:
      ValueError: if the image is not able to be found. This indicates the file 
      structure of the Pix3D folder is incorrect.
    """

    with tf.io.gfile.GFile(
        os.path.join(image["pix3d_dir"], image["img"]), 'rb') as fid:
            encoded_img = fid.read()

    img_width, img_height = image["img_size"]
    img_filename = image["img"]
    img_category = image["category"]
    keypoints_2d = image["2d_keypoints"]

    feature_dict = {"img/height": convert_to_feature(img_height),
                    "img/width": convert_to_feature(img_width),
                    "img/category": convert_to_feature(img_category),
                    "img/filename": convert_to_feature(img_filename),
                    "img/encoded": convert_to_feature(encoded_img),
                    "img/2d_keypoints": convert_to_feature(keypoints_2d)}

    with tf.io.gfile.GFile(os.path.join(image["pix3d_dir"], image["mask"]), 'rb') as fid:
        encoded_mask = fid.read()

    feature_dict.update({"mask": convert_to_feature(encoded_mask)})

    model_vertices, model_faces = parse_obj_file(
        os.path.join(image["pix3d_dir"], image["model"]))

    with tf.io.gfile.GFile(
        os.path.join(image["pix3d_dir"], image["3d_keypoints"]), 'rb') as fid:
            keypoints_3d = fid.read()

    model_raw = image["model_raw"]
    model_source = image["model_source"]

    feature_dict.update(
        {"model/vertices": convert_to_feature(model_vertices),
         "model/faces": convert_to_feature(model_faces),
         "model/raw": convert_to_feature(model_raw),
         "model/source": convert_to_feature(model_source),
         "model/3d_keypoints": convert_to_feature(keypoints_3d)})

    with tf.io.gfile.GFile(
        os.path.join(image["pix3d_dir"], image["voxel"]), 'rb') as fid:
            encoded_voxel = fid.read()

    rot_mat = image["rot_mat"]
    trans_mat = image["trans_mat"]
    focal_length = image["focal_length"]
    cam_position = image["cam_position"]
    inplane_rotation = image["inplane_rotation"]
    truncated = image["truncated"]
    occluded = image["occluded"]
    slightly_occluded = image["slightly_occluded"]
    bbox = image["bbox"]

    feature_dict.update(
        {"voxel": convert_to_feature(encoded_voxel),
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

def parse_obj_file(file):
    """
    Parses relevant data out of a .obj file.
    Args:
        file: file path to .obj file
    Return:
        vertices: vertices of object
        faces: faces of object
    """
    vertices = []
    faces = []

    obj_file = open(file, 'r')
    lines = obj_file.readlines()

    for line in lines:
        lineID = line[0:2]

        if lineID == "v ":
            vertex = line[2:].split(" ")
            for i, v in enumerate(vertex):
                vertex[i] = float(v)
            vertices.append(vertex)

        if lineID == "f ":
            face = line[2:].split(" ")
            for i, f in enumerate(face):
                face[i] = [int(x) for x in f.split("/")]
            faces.append(face)

    return vertices, faces


def generate_annotations(images, pix3d_dir):
    """Generator for Pix3D annotations."""

    annotations = []

    for image in images:
        annotations.append(
            {"img": image["img"], "category": image["category"], 
            "img_size": image["img_size"], 
            "2d_keypoints": image["2d_keypoints"],
            "mask": image["mask"], "img_source": image["img_source"], 
            "model": image["model"], "model_raw": image["model_raw"],
            "model_source": image["model_source"], 
            "3d_keypoints": image["3d_keypoints"], "voxel": image["voxel"], 
            "rot_mat": image["rot_mat"],
            "trans_mat": image["trans_mat"], 
            "focal_length": image["focal_length"], 
            "cam_position": image["cam_position"],
            "inplane_rotation": image["inplane_rotation"], 
            "truncated": image["truncated"], "occluded": image["occluded"],
            "slightly_occluded": image["slightly_occluded"], 
            "bbox": image["bbox"], "pix3d_dir": pix3d_dir})

    return annotations


def _create_tf_record_from_pix3d_dir(pix3d_dir,
                                     output_path,
                                     num_shards,
                                     pix3d_json_file):
    """Loads Pix3D json files and converts to tf.Record format.
    Args:
      images_info_file: pix3d_dir download directory
      output_path: Path to output tf.Record file.
      num_shards: Number of output files to create.
    """

    logging.info('writing to output path: %s', output_path)

    images = json.load(open(os.path.join(pix3d_dir, pix3d_json_file)))

    pix3d_annotations_iter = generate_annotations(
        images=images, pix3d_dir=pix3d_dir)

    num_skipped = write_tf_record_dataset(
        output_path, pix3d_annotations_iter, create_tf_example, 
        num_shards, unpack_arguments=False, use_multiprocessing=True)

    logging.info('Finished writing, skipped %d annotations.', num_skipped)

def main(_):
    assert FLAGS.pix3d_dir, '`pix3d_dir` missing.'

    directory = os.path.dirname(FLAGS.output_file_prefix)
    if not tf.io.gfile.isdir(directory):
        tf.io.gfile.makedirs(directory)

    _create_tf_record_from_pix3d_dir(
        FLAGS.pix3d_dir[0], FLAGS.output_file_prefix, FLAGS.num_shards, 
        FLAGS.pix3d_json_file[0])

if __name__ == '__main__':
    app.run(main)
