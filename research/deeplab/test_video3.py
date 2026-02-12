###  COPY ALL THE CODE INTO A JYPYTER NOTEBOOK  ###
###  THE JYPYTER NOTEBOOK NEEDS TO BE IN 'tensorflow\models\research\deeplab'  ###

# Imports
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
import get_dataset_colormapTomas
# Needed to show segmentation colormap labels


# print(sys.path)


# Load model in TensorFlow

_FROZEN_GRAPH_NAME = 'frozen_inference_graph'
plt=pyplot
plt1=pyplot
plt2=pyplot
plt3=pyplot
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
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
            image: A PIL.Image object, raw input image.

        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0# * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
    def run_video(self, video_path):
        """Runs inference on a single video.

        Args:
        path: Path to video

        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        file, ext = os.path.splitext(video_path)
        video_name = file.split('/')[-1]
        out_filename = video_name + '_SEG_bez_prevodu_farieb' + '.avi'

        cap = cv2.VideoCapture(video_path)
        wi = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        he = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(wi, he)

        vwriter = cv2.VideoWriter(out_filename,cv2.VideoWriter_fourcc(*'MJPG'),30, (wi, he))
        counter = 0
        fac = 2
        start = time.time()
        while True:
            ret, image = cap.read()
            if ret:
                counter += 1
                print(counter)
                ## resize image

                height, width, channels = image.shape
                resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
                target_size = (int(resize_ratio * width), int(resize_ratio * height))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
                output = resized_image.copy()

                ## get segmentation map
                batch_seg_map = self.sess.run(
                self.OUTPUT_TENSOR_NAME,
                feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
                seg_map = batch_seg_map[0]

                ## visualize
                #povodne seg_image = label_to_color_image(seg_map).astype(np.uint8)
                #doplnene:
                #seg_image = get_dataset_colormap.label_to_color_image(seg_map, get_dataset_colormap.get_rugd_name()).astype(np.uint8)
                seg_image = get_dataset_colormapTomas.label_to_color_image(seg_map, get_dataset_colormapTomas.get_rugd_name()).astype(np.uint8)
                seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2RGB)
                ## overlay on image
                alpha = 0.7
                cv2.addWeighted(seg_image, alpha, output, 1 - alpha, 0, output)

                output = cv2.resize(output, (wi, he), interpolation=cv2.INTER_AREA)
                #outimg = 'image_' + str(counter) + '.jpg'
                #cv2.imwrite(os.path.join(os.getcwd(), 'test_out', outimg),output)
                vwriter.write(output)
            else:
                break
        end = time.time()
        print("Frames and Time Taken: ", counter, end-start)
        cap.release()
        vwriter.release()   


download_path = os.path.dirname(__file__)+'\\datasets\\RUGD\\exp\\train_on_trainval_set\\export\\frozen_inference_graph.pb'
model = DeepLabModel(download_path)

LABEL_NAMES = np.asarray(['container/generic-object',  'tree', 'water', 'sky', 'rock', 'dirt', 'sand', 'grass', 'vehicle', 'bush', 'bridge', 'rock-bed',
                          'picnic-table', 'concrete', 'sign', 'fence', 'person', 'bicycle', 'log', 'mulch', 'building', 'gravel', 'asphalt', 'pole', 'void',
                          ])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = get_dataset_colormapTomas.label_to_color_image(FULL_LABEL_MAP)


def vis_segmentation(image, seg_map):
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6,6,6, 1])
    seg_image = get_dataset_colormapTomas.label_to_color_image(seg_map, get_dataset_colormapTomas.get_rugd_name()).astype(np.uint8)
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
    
    seg_image = get_dataset_colormapTomas.label_to_color_image(seg_map, get_dataset_colormapTomas.get_rugd_name()).astype(np.uint8)
    
    plt1.figure(figsize=(10,10))
    plt1.imshow(image)
    plt1.axis('off')
    plt1.title('input image')

    
    plt2.figure(figsize=(10,10))
    plt2.imshow(seg_image)
    plt2.axis('off')
    plt2.title('segmentation map')
    
    #plt2.savefig(os.path.dirname(__file__)+'\\datasets\\rugd\\ObrazkyExport\Obrazok.png')

    plt3.figure(figsize=(10,10))
    plt3.imshow(image)
    plt3.imshow(seg_image, alpha=0.7)
    plt3.axis('off')
    plt3.title('segmentation overlay')

    plt1.show()
    plt2.show()
    plt3.show() 

IMAGE_DIR = os.path.dirname(__file__)+'\\datasets\\RUGD\\Skolsky_dataset\\'

    
def run_demo_image(image_name,i):
    try:
        image_path = os.path.join(IMAGE_DIR, image_name)
        original_im = Image.open(image_path)
    except IOError:
        print('Failed to read image from %s.' % image_path)
        return
    print('running deeplab on image %s...' % image_name)
    resized_im, seg_map = model.run(original_im)
    #video
    # model.run_video(os.path.dirname(__file__)+'\\datasets\\RUGD\\TestovacieObrazky\\output.mp4')
    #1 image visualization
    #vis_segmentation(resized_im, seg_map)

    #saving as image
    seg_map = get_dataset_colormapTomas.label_to_color_image(seg_map, get_dataset_colormapTomas.get_rugd_name()).astype(np.uint8)
    txt = "{}".format(i).zfill(5)+'.bmp'
    #print(txt)
    plt2.imsave(os.path.dirname(__file__)+'\\datasets\\RUGD\\ObrazkyExport\\'+txt, seg_map)

#run_demo_image('creek_00001.png')
#run_demo_image('Lapaj.jpg')
#run_demo_image('Clipboard07.jpg')
#run_demo_image('IMG20230401143409.jpg')
for i in range(8849,8851):
    txt = "{}".format(i).zfill(5)
    #print(txt+'.png')
    run_demo_image(txt+'.png',i)
#run_demo_image('IMG_20210510_092303.jpg')
#run_demo_image('IMG_20210510_092303_1.jpg')
#run_demo_image('IMG_20210510_092304.jpg')