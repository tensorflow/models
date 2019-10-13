import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from PIL import Image
import json, argparse, time
from flask import Flask, request
from flask_cors import CORS

from DeepLabV3 import DeepLabModel
from six.moves import urllib
from matplotlib import gridspec
from matplotlib import pyplot as plt


download_path1 = "../model/deeplabv3_xception_ade20k_train_2018_05_29.tar.gz"
download_path2 = "../model/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz"
saved_model_path = '../model/deeplab_ade20k/10/'

#MODEL = DeepLabModel(download_path1)
#MODEL.store(saved_model_path)
#resized_img, res =MODEL.runWithCV('./data/ki_corridor/1.png')
#plt.figure(figsize=(20, 15))
#plt.imshow(res)
#plt.show()


##################################################
# API part
##################################################
app = Flask(__name__)
cors = CORS(app)
@app.route("/api/predict", methods=['POST'])
def predict():
    start = time.time()

    data = request.data.decode("utf-8")

    if data == "":
        params = request.form
        x_in = json.loads(params['image'])

    else:
        params = json.loads(data)
        print(params)
        x_in = params['image']
        print(type(x_in))

    ############
    #Convert PIL Image
    ######
    width = len(x_in[0])
    height = len(x_in)
    print(type(x_in), width, height)

    na = np.array(x_in, dtype=np.uint8)
    print(na.shape)


    img = Image.fromarray(na, 'RGB')

    img.save('./data/target.jpg')
    im = img.load()
    print(img.width, img.height)
    print(im[0,0], na[0][0])
    print(im[1,0], na[1][0])
    print(im[2,0], na[2][0])
    #img = Image.new('RGB', (width, height))
    #img.putdata(tuple(x_in))

    #plt.figure(figsize=(20, 15))
    #plt.imshow(img)
    #plt.show()

    ##################################################
    # Tensorflow part
    ##################################################
    resized_img, seg_map = MODEL.run(img)


    #y_out = persistent_sess.run(y, feed_dict={
    #    x: x_in
    #})
    ##################################################
    # END Tensorflow part
    ##################################################

    json_data = json.dumps({'y': seg_map.tolist()})
    print("Time spent handling the request: %f" % (time.time() - start))

    return json_data
##################################################
# END API part
##################################################

if __name__ == "__main__":

    ##################################################
    # Tensorflow part
    ##################################################
    MODEL = DeepLabModel(download_path1)
    graph = MODEL.graph
    x = graph.get_tensor_by_name(MODEL.INPUT_TENSOR_NAME)
    y = graph.get_tensor_by_name(MODEL.OUTPUT_TENSOR_NAME)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)
    persistent_sess = tf.Session(graph=graph, config=sess_config)
    ##################################################
    # END Tensorflow part
    ##################################################

    print('Starting the API')
    app.run(host='143.248.96.81', port = 35005)