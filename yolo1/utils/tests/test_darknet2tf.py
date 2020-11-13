from absl.testing import parameterized
import tensorflow as tf
import numpy as np

try:
    from importlib import resources as importlib_resources
except BaseException:
    # Shim for Python 3.6 and older
    import importlib_resources

from official.vision.beta.projects.yolo.modeling.backbones.darknet import Darknet
from yolo.utils._darknet2tf import DarkNetConverter
from yolo.utils._darknet2tf.load_weights2 import load_weights_backbone

class darknet2tf_test(tf.test.TestCase, parameterized.TestCase):
    def test_load_yolov3_weights(self):
        x = tf.ones(shape=[1, 224, 224, 3], dtype=tf.float32)
        model = Darknet(model_id='darknettiny')
        encoder = DarkNetConverter.read('cache/cfg/yolov3-tiny.cfg', 'cache/weights/yolov3-tiny.weights')
        encode = encoder[:12]
        load_weights_backbone(model, encoder)
        y: tf.Tensor = model(x)

if __name__ == "__main__":
    tf.test.main()
