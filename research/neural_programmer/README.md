![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Neural Programmer

Implementation of the Neural Programmer model as described in this [paper](https://openreview.net/pdf?id=ry2YOrcge).

Download and extract the data from the [WikiTableQuestions](https://ppasupat.github.io/WikiTableQuestions/) site. The dataset contains
11321, 2831, and 4344 examples for training, development, and testing respectively. We use their tokenization, number and date pre-processing. Please note that the above paper used the [initial release](https://github.com/ppasupat/WikiTableQuestions/releases/tag/v0.2) for training, development and testing. 

Change the `data_dir FLAG` to the location of the data.

### Training 
Run `python neural_programmer.py` 

The models are written to `FLAGS.output_dir`.

### Testing 
Run `python neural_programmer.py --evaluator_job=True`

The models are loaded from `FLAGS.output_dir`. The evaluation is done on development data.

In case of errors because of encoding, add `"# -*- coding: utf-8 -*-"` as the first line in `wiki_data.py`

Maintained by Arvind Neelakantan (arvind2505)
