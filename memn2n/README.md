End-To-End Memory Networks in Tensorflow
========================================

Tensorflow implementation of [End-To-End Memory Networks](http://arxiv.org/abs/1503.08895v4) for language modeling (see Section 5). The original torch code from Facebook can be found [here](https://github.com/facebook/MemNN/tree/master/MemN2N-lang-model).


Prerequisites
-------------

This code requires [Tensorflow](https://www.tensorflow.org/). There is a set of sample Penn Tree Bank (PTB) corpus in `data` directory, which is a popular benchmark for measuring quality of these models. But you can use your own text data set which should be formated like [this](data/).


Usage
-----

To train a model with 6 hops and memory size of 100, run the following command:

    $ python main.py --nhop 6 --mem_size 100

To see all training options, run:

    $ python main.py --help

which will print:

    usage: main.py [-h] [--edim EDIM] [--lindim LINDIM] [--nhop NHOP]
                  [--mem_size MEM_SIZE] [--batch_size BATCH_SIZE]
                  [--nepoch NEPOCH] [--init_lr INIT_LR] [--init_hid INIT_HID]
                  [--init_std INIT_STD] [--max_grad_norm MAX_GRAD_NORM]
                  [--data_dir DATA_DIR] [--data_name DATA_NAME] [--show SHOW]
                  [--noshow]

    optional arguments:
      -h, --help            show this help message and exit
      --edim EDIM           internal state dimension [150]
      --lindim LINDIM       linear part of the state [75]
      --nhop NHOP           number of hops [6]
      --mem_size MEM_SIZE   memory size [100]
      --batch_size BATCH_SIZE
                            batch size to use during training [128]
      --nepoch NEPOCH       number of epoch to use during training [100]
      --init_lr INIT_LR     initial learning rate [0.01]
      --init_hid INIT_HID   initial internal state value [0.1]
      --init_std INIT_STD   weight initialization std [0.05]
      --max_grad_norm MAX_GRAD_NORM
                            clip gradients to this norm [50]
      --checkpoint_dir CHECKPOINT_DIR
                            checkpoint directory [checkpoints]
      --data_dir DATA_DIR   data directory [data]
      --data_name DATA_NAME
                            data set name [ptb]
      --is_test IS_TEST     True for testing, False for Training [False]
      --nois_test
      --show SHOW           print progress [False]
      --noshow

(Optional) If you want to see a progress bar, install `progress` with `pip`:

    $ pip install progress
    $ python main.py --nhop 6 --mem_size 100 --show True

After training is finished, you can test and validate with:

    $ python main.py --is_test True --show True


Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
