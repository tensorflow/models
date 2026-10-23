This Docker image automates the setup involved with training
object detection models on Google Cloud and building TensorFlow Lite models

A word of warning:

-- Docker containers do not have persistent storage. This means that any changes
   you make to files inside the container will not persist if you restart
   the container. When running through the tutorial,
   **do not close the container**.

You can install Docker by following the [instructions here](
https://docs.docker.com/install/).

## Running The Container

From this directory, build the Dockerfile as follows (this takes a while):

```
docker build --tag detect-tf-generic .
```

Run the container:

```
docker run --rm -it --privileged -p 6006:6006 detect-tf-generic
```

When running the container, you will find yourself inside the `/tensorflow`
directory, which is the path to the workspace of this docker container

This Docker images comes with `vim`, `nano`, and `emacs` preinstalled for your
convenience.

## What's In This Container

This container is derived from the v1.15.0 build of TensorFlow, and contains the
[TensorFlow Models](https://github.com/tensorflow/models) repo available at
`/tensorflow/models` (and contain the Object Detection API as a subdirectory
at `/tensorflow/models/research/object_detection`).

This container also has the `gsutil` and `gcloud` utilities and all dependencies necessary to use the Object Detection API
