# Dockerfile for the TPU and TensorFlow Lite Object Detection tutorial

This Docker image automates the setup involved with training
object detection models on Google Cloud and building the Android TensorFlow Lite
demo app. We recommend using this container if you decide to work through our
tutorial on ["Training and serving a real-time mobile object detector in
30 minutes with Cloud TPUs"](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193), though of course it may be useful even if you would
like to use the Object Detection API outside the context of the tutorial.

A couple words of warning:

1. Docker containers do not have persistent storage. This means that any changes
   you make to files inside the container will not persist if you restart
   the container. When running through the tutorial,
   **do not close the container**.
2. To be able to deploy the [Android app](
   https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples/android/app)
   (which you will build at the end of the tutorial),
   you will need to kill any instances of `adb` running on the host machine. You
   can accomplish this by closing all instances of Android Studio, and then
   running `adb kill-server`.

You can install Docker by following the [instructions here](
https://docs.docker.com/install/).

## Running The Container

From this directory, build the Dockerfile as follows (this takes a while):

```
docker build --tag detect-tf .
```

Run the container:

```
docker run --rm -it --privileged -p 6006:6006 detect-tf
```

When running the container, you will find yourself inside the `/tensorflow`
directory, which is the path to the TensorFlow [source
tree](https://github.com/tensorflow/tensorflow).

## Text Editing

The tutorial also
requires you to occasionally edit files inside the source tree.
This Docker images comes with `vim`, `nano`, and `emacs` preinstalled for your
convenience.

## What's In This Container

This container is derived from the nightly build of TensorFlow, and contains the
sources for TensorFlow at `/tensorflow`, as well as the
[TensorFlow Models](https://github.com/tensorflow/models) which are available at
`/tensorflow/models` (and contain the Object Detection API as a subdirectory
at `/tensorflow/models/research/object_detection`).
The Oxford-IIIT Pets dataset, the COCO pre-trained SSD + MobileNet (v1)
checkpoint, and example
trained model are all available in `/tmp` in their respective folders.

This container also has the `gsutil` and `gcloud` utilities, the `bazel` build
tool, and all dependencies necessary to use the Object Detection API, and
compile and install the TensorFlow Lite Android demo app.

At various points throughout the tutorial, you may see references to the
*research directory*.  This refers to the `research` folder within the
models repository, located at
`/tensorflow/models/resesarch`.
