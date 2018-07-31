# Deploying a tf.keras model with tf.serving - A temperature forecasting example

In order to run this code you will need the following prerequisites:
* h5py - `pip install h5py`
* [TensorFlow](https://www.tensorflow.org/install/) - `pip install TensorFlow`
* [TensorFlow Serving pip package](https://www.tensorflow.org/serving/setup#tensorflow_serving_python_api_pip_package) - `pip install tensorflow-serving-api`
* [These Prerequisites](https://www.tensorflow.org/serving/setup#prerequisites)

For launching the server, you will either need:
* [Bazel](https://www.tensorflow.org/serving/setup#bazel_only_if_compiling_source_code)

OR 

* [Install TensorFlow Serving ModelServer with apt-get](https://www.tensorflow.org/serving/setup#installing_using_apt_get)