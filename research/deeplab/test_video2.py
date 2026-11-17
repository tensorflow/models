###  COPY ALL THE CODE INTO A JYPYTER NOTEBOOK  ###
###  THE JYPYTER NOTEBOOK NEEDS TO BE IN 'tensorflow\models\research\deeplab'  ###

# Imports
import get_dataset_colormap
import collections
import os
import io
import sys
import tarfile
import tempfile
import urllib

from IPython import display
from ipywidgets import interact
from ipywidgets import interactive
from matplotlib import gridspec
from matplotlib import pyplot
import numpy as np
from PIL import Image
import cv2
import time

import tensorflow as tf

sys.path.append(os.path.dirname(__file__) + "\\utils")

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'
plt = pyplot
plt1 = pyplot
plt2 = pyplot
plt3 = pyplot


class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, graph_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        with tf.io.gfile.GFile(graph_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run_pic(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def run_video(self, video_path):

        if(os.path.isfile(video_path)):
            file_name, ext = os.path.splitext(video_path)

            out_filename = file_name + '_SEG_bez_prevodu_farieb' + '.avi'

            cap = cv2.VideoCapture(video_path)
            while cap.isOpened():
                wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                he = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(wi, he)

                vwriter = cv2.VideoWriter(out_filename, cv2.VideoWriter_fourcc(*'MJPG'), 30, (wi, he))
                counter = 0
                fac = 2
                start = time.time()
                while True:
                    ret, image = cap.read()
                    if not ret:
                        print("Can't receive frame (stream end?). Exiting ...")
                        break
                    if ret:
                        # resize image
                        '''
                        height, width, channels = image.shape
                        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
                        target_size = (int(resize_ratio * width), int(resize_ratio * height))
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                        output = resized_image.copy()
                        '''
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        resized_image = image
                        output = resized_image.copy()
                        if counter % 100 == 0:
                            print(counter, time.time()-start)
                        if counter % SKIP_PICTURES == 0:
                            # get segmentation map
                            batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME, feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
                            seg_map = batch_seg_map[0]

                            # visualize
                            seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_rugd_name()).astype(np.uint8)
                            seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)

                        alpha = 0.7
                        cv2.addWeighted(seg_image, alpha, output, 1 - alpha, 0, output)

                        output = cv2.resize(output, (wi, he), interpolation=cv2.INTER_AREA)
                        vwriter.write(output)
                        counter += 1
                end = time.time()
                print("Frames and Time Taken: ", counter, end-start)
                cap.release()
                vwriter.release()
                cv2.destroyAllWindows()


def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
    seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_rugd_name()).astype(np.uint8)
    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.3)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0)

    plt.show()


def vis_segmentation2(image, seg_map):

    seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_rugd_name()).astype(np.uint8)

    plt1.figure(figsize=(10, 10))
    plt1.imshow(image)
    plt1.axis('off')
    plt1.title('input image')

    plt2.figure(figsize=(10, 10))
    plt2.imshow(seg_image)
    plt2.axis('off')
    plt2.title('segmentation map')

    plt2.savefig(os.path.dirname(__file__)+'\\datasets\\rugd\\ObrazkyExport\Obrazok.png')

    plt3.figure(figsize=(10, 10))
    plt3.imshow(image)
    plt3.imshow(seg_image, alpha=0.7)
    plt3.axis('off')
    plt3.title('segmentation overlay')

    plt1.show()
    plt2.show()
    plt3.show()


frozen_inference_graph_path = os.path.dirname(__file__)+'\\datasets\\rugd\exp\\train_on_trainval_set\\export\\frozen_inference_graph.pb'
model = DeepLabModel(frozen_inference_graph_path)

LABEL_NAMES = np.asarray(['container/generic-object',  'tree', 'water', 'sky', 'rock', 'dirt', 'sand', 'grass', 'vehicle', 'bush', 'bridge', 'rock-bed',
                          'picnic-table', 'concrete', 'sign', 'fence', 'person', 'bicycle', 'log', 'mulch', 'building', 'gravel', 'asphalt', 'pole', 'void',
                          ])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormap.label_to_color_image(FULL_LABEL_MAP)
IMAGE_DIR = os.path.dirname(__file__)+'\\datasets\\RUGD\\TestovacieObrazky\\'
SKIP_PICTURES = 5

model.run_video('output.mp4')
