# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""Model inference function for object detection model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
import glob
import argparse
from PIL import Image
from utils import label_map_util
from utils import visualization_utils as vis_util
from utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.0!')


parser = argparse.ArgumentParser()

parser.add_argument("--input_dir", help="Path of the input images directory")
parser.add_argument("--frozen_graph", help="Path of the frozen graph model")
parser.add_argument("--label_map", help="Path of the label map file")
parser.add_argument("--output_dir", help="Path of the output directory")
parser.add_argument("--num_output_classes", default=90,
                    help="Defines the number of output classes", type=int)

args = parser.parse_args()
PATH_TO_CKPT = args.frozen_graph
PATH_TO_LABELS = args.label_map
NUM_CLASSES = args.num_output_classes
PATH_TO_TEST_IMAGES_DIR = args.input_dir
PATH_TO_RESULT_IMAGES_DIR = args.output_dir


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = \
                        tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(
                    tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(
                    tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates
                # to image coordinates and fit the image size.
                real_num_detection = tf.cast(
                    tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                           real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                           real_num_detection, -1, -1])
                detection_masks_reframed = \
                    utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks,
                        detection_boxes,
                        image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = \
                tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays,
            # so convert types as appropriate
            output_dict['num_detections'] = int(
                output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def main(_):
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    TEST_IMAGE_PATHS = glob.glob(
        os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

    JPG_PATHS = [os.path.basename(path) for path in TEST_IMAGE_PATHS]

    RESULT_IMAGE_PATHS = [os.path.join(
        PATH_TO_RESULT_IMAGES_DIR, jpg_path) for jpg_path in JPG_PATHS]

    # Load a (frozen) Tensorflow model into memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Loading label map
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    count = 0
    for image_path, result_path in \
            zip(TEST_IMAGE_PATHS, RESULT_IMAGE_PATHS):
        image = Image.open(image_path)
        # the array based representation of the image will be used later
        # in order to prepare the result image with boxes and labels on it.
        image_np = load_image_into_numpy_array(image)
        # Actual detection.
        output_dict = run_inference_for_single_image(
            image_np, detection_graph)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)
        im = Image.fromarray(image_np)
        im.save(result_path)
        count += 1
        print('Images Processed:', count, end='\r')


if __name__ == '__main__':
    tf.app.run()
