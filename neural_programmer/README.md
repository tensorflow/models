Implementation of the Neural Programmer model described in https://openreview.net/pdf?id=ry2YOrcge

Download and extract the data from https://www.dropbox.com/s/9tvtcv6lmy51zfw/data.zip?dl=0. Change the data_dir FLAG to the location of the data.

Training: python neural_programmer.py. The models are written to FLAGS.output_dir

Testing: python neural_programmer.py --evaluator_job=True. The models are loaded from FLAGS.output_dir. The evaluation is done on development data.

In case of errors because of encoding, add "# -*- coding: utf-8 -*-" as the first line in wiki_data.py

Maintained by Arvind Neelakantan (arvind2505)
