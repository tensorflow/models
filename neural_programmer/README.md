Implementation of the Neural Programmer model described in https://openreview.net/pdf?id=ry2YOrcge

Download the data from http://www-nlp.stanford.edu/software/sempre/wikitable/ Change the data_dir FLAG to the location of the data

Training: python neural_programmer.py

The models are written to FLAGS.output_dir

Testing: python neural_programmer.py --evaluator_job=True

The models are loaded from FLAGS.output_dir. The evaluation is done on development data.

Maintained by Arvind Neelakantan (arvind2505)
