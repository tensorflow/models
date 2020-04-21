![No Maintenance Intended](https://img.shields.io/badge/No%20Maintenance%20Intended-%E2%9C%95-red.svg)
![TensorFlow Requirement: 1.x](https://img.shields.io/badge/TensorFlow%20Requirement-1.x-brightgreen)
![TensorFlow 2 Not Supported](https://img.shields.io/badge/TensorFlow%202%20Not%20Supported-%E2%9C%95-red.svg)

# Neural Programmer

Implementation of the Neural Programmer model described in [paper](https://openreview.net/pdf?id=ry2YOrcge)

Download and extract the data from [dropbox](https://www.dropbox.com/s/9tvtcv6lmy51zfw/data.zip?dl=0). Change the ``data_dir FLAG`` to the location of the data.

### Training 
``python neural_programmer.py`` 

The models are written to FLAGS.output_dir

### Testing 
``python neural_programmer.py --evaluator_job=True``

The models are loaded from ``FLAGS.output_dir``. The evaluation is done on development data.

In case of errors because of encoding, add ``"# -*- coding: utf-8 -*-"`` as the first line in ``wiki_data.py``

Maintained by Arvind Neelakantan (arvind2505)
