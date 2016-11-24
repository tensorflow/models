# TensorFlow Models

This repository contains machine learning models implemented in
[TensorFlow](https://tensorflow.org). The models are maintained by their
respective authors.

To propose a model for inclusion please submit a pull request.


## Models
- [autoencoder](autoencoder) -- various autoencoders
- [inception](inception) -- deep convolutional networks for computer vision
- [namignizer](namignizer) -- recognize and generate names
- [neural_gpu](neural_gpu) -- highly parallel neural computer
- [privacy](privacy) -- privacy-preserving student models from multiple teachers
- [resnet](resnet) -- deep and wide residual networks
- [slim](slim) -- image classification models in TF-Slim
- [swivel](swivel) -- the Swivel algorithm for generating word embeddings
- [syntaxnet](syntaxnet) -- neural models of natural language syntax
- [textsum](textsum) -- sequence-to-sequence with attention model for text summarization.
- [transformer](transformer) -- spatial transformer network, which allows the spatial manipulation of data within the network
- [im2txt](im2txt) -- image-to-text neural network for image captioning.
=======
Implementation of the Neural Programmer model described in https://openreview.net/pdf?id=ry2YOrcge

Download the data from http://www-nlp.stanford.edu/software/sempre/wikitable/
Change the data_dir FLAG to the location of the data

Training:
python neural_programmer.py

The models are written to FLAGS.output_dir 


Testing:
python neural_programmer.py --evaluator_job=True

The models are loaded from FLAGS.output_dir.
The evaluation is done on development data.

Maintained by Arvind Neelakantan (arvind2505)
