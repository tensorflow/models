#!/usr/bin/env python

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import pickle
import argparse

import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage import img_as_ubyte
import tensorflow as tf
from tensorflow.contrib.graph_editor import reroute


MODEL_DIR = 'model'
KEY_FN = 'key.pkl'
MODEL_FN = 'mobilenet_v1_1.0_224/frozen_graph.pb'

MODEL_INFO = {
    'input_name': 'input:0',
    'output_name': 'MobilenetV1/Predictions/Softmax:0',
    'logits_name': 'MobilenetV1/Logits/SpatialSqueeze:0'
}


def main(args):
    # Load the keys so we can print human readable labels.
    with open(os.path.join(MODEL_DIR, KEY_FN), 'r') as kf:
        key = pickle.load(kf)

    original = plt.imread(args.img_fn)
    a = resize(original, (224, 224, 3), mode='constant')

    # Load the frozen graph into the default graph.
    with open(os.path.join(MODEL_DIR, MODEL_FN), 'rb') as f:
        graph_def = tf.GraphDef.FromString(f.read())

    input_, logits, prob = tf.import_graph_def(
        graph_def, name='',
        return_elements=[MODEL_INFO['input_name'], MODEL_INFO['logits_name'], MODEL_INFO['output_name']])

    # Create a variable and reroute from it.
    var_input = tf.get_variable(name='var_input', dtype=input_.dtype, shape=input_.shape)
    reroute._reroute_t(var_input.value(), input_, input_.consumers())

    # Instead of feeding data, we would assign the data to the var_input
    # variable when we want to specify an image.
    data = tf.placeholder(dtype=var_input.dtype, shape=var_input.shape)
    assign = tf.assign(var_input, data)

    # Get the highest softmax score and the label
    score = tf.reduce_max(prob)
    index = tf.argmax(prob, axis=1)[0]

    # Feed label based on index during session.
    one_hot = tf.one_hot([index], 1001)
    label = tf.placeholder(dtype=tf.float32, shape=logits.shape)

    # The loss function is to minimize the top-1 label's score.
    loss = tf.losses.softmax_cross_entropy(label, logits)

    # Calculate and report entropy to get a sense of the model's
    # uncertainty.
    entropy = - tf.reduce_sum(prob * tf.log(prob))

    # Using an optimizer to calculate gradients.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    grad, var = optimizer.compute_gradients(loss, var_list=[var_input])[0]

    with tf.Session() as sess:
        for i in range(args.n_iter):
            # specify an image to perturb
            _ = sess.run(assign, {data: [a]})

            _one_hot = sess.run(one_hot)
            _grad = sess.run(grad, {label: _one_hot})[0]

            print(sess.run(score), key[sess.run(index)], sess.run(entropy))

            # compute sign, then update
            _grad[_grad > 0] = 1.0
            _grad[_grad < 0] = -1.0
            a = a + args.epsilon * _grad

        # final image
        _ = sess.run(assign, {data: [a]})
        print(sess.run(score), key[sess.run(index)], sess.run(entropy))

        # save the images
        original_img = Image.fromarray(img_as_ubyte(original))
        original_img.save('original.png')

        _grad_resized = resize(_grad, list(original.shape)[:2], mode='constant')
        _grad_out = img_as_ubyte(_grad_resized)
        _grad_img = Image.fromarray(_grad_out)
        _grad_img.save('signg.png')

        a_modified = sess.run(var_input)[0]
        a_modified[a_modified > 1] = 1
        a_modified[a_modified < -1] = -1
        a_resized = resize(a_modified, list(original.shape)[:2], mode='constant')
        a_out = img_as_ubyte(a_resized)
        img = Image.fromarray(a_out)
        img.save('out.png')


    from subprocess import call
    call(['open', 'original.png', 'signg.png', 'out.png'])
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.007)
    parser.add_argument('--n-iter', type=int, default=1)
    parser.add_argument('img_fn')

    args = parser.parse_args()

    main(args)

    



