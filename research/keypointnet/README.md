![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# KeypointNet
This is an implementation of the keypoint network proposed in "Discovery of
Latent 3D Keypoints via End-to-end Geometric Reasoning
[[pdf](https://arxiv.org/pdf/1807.03146.pdf)]". Given a single 2D image of a
known class, this network can predict a set of 3D keypoints that are consistent
across viewing angles of the same object and across object instances. These
keypoints and their detectors are discovered and learned automatically without
keypoint location supervision [[demo](https://keypointnet.github.io)].

## Datasets:
  ShapeNet's rendering for 
  [Cars](https://storage.googleapis.com/discovery-3dkeypoints-data/cars_with_keypoints.zip),
  [Planes](https://storage.googleapis.com/discovery-3dkeypoints-data/planes_with_keypoints.zip),
  [Chairs](https://storage.googleapis.com/discovery-3dkeypoints-data/chairs_with_keypoints.zip).

  Each set contains:
1. tfrecords
2. train.txt, a list of tfrecords used for training.
2. dev.txt, a list of tfrecords used for validation.
3. test.txt, a list of tfrecords used for testing.
4. projection.txt, storing the global 4x4 camera projection matrix.
5. job.txt, storing ShapeNet's object IDs in each tfrecord.
  
## Training:
  Run `main.py --model_dir=MODEL_DIR --dset=DSET`

  where MODEL_DIR is a folder for storing model checkpoints: (see [tf.estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)), and DSET should point to the folder containing tfrecords (download above).

## Inference:
  Run `main.py --model_dir=MODEL_DIR --input=INPUT --predict`

  where MODEL_DIR is the model checkpoint folder, and INPUT is a folder containing png or jpeg test images.
  We trained the network using the total batch size of 256 (8 x 32 replicas). You may have to tune the learning rate if your batch size is different. 

## Code credit:
  Supasorn Suwajanakorn

## Contact:
  supasorn@gmail.com, [snavely,tompson,mnorouzi]@google.com


(This is not an officially supported Google product)
