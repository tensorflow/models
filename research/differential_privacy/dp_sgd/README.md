<font size=4><b>Deep Learning with Differential Privacy</b></font>

Authors:
Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang

Open Sourced By: Xin Pan (xpan@google.com, github: panyx0718)


<Introduction>

Machine learning techniques based on neural networks are achieving remarkable
results in a wide variety of domains. Often, the training of models requires
large, representative datasets, which may be crowdsourced and contain sensitive
information. The models should not expose private information in these datasets.
Addressing this goal, we develop new algorithmic techniques for learning and a
refined analysis of privacy costs within the framework of differential privacy.
Our implementation and experiments demonstrate that we can train deep neural
networks with non-convex objectives, under a modest privacy budget, and at a
manageable cost in software complexity, training efficiency, and model quality.

paper: https://arxiv.org/abs/1607.00133


<b>Requirements:</b>

1. Tensorflow 1.4.0 (master branch)

2. Bazel 0.5.4

3. Download MNIST data

<b>How to run:</b>

```shell
# Clone the code
# Create an empty WORKSPACE file, eg.
$ cd models/research
$ touch WORKSPACE

# Download the data to the data/ directory.
$ cd models/research/slim/
$ DATA_DIR=models/research/data
$ python download_and_convert_data.py --dataset_name=mnist --dataset_dir="${DATA_DIR}"

# From models/research, list the codes.
$ ls -R differential_privacy/
differential_privacy/:
dp_sgd  __init__.py  privacy_accountant  README.md

differential_privacy/dp_sgd:
dp_mnist  dp_optimizer  per_example_gradients  README.md

differential_privacy/dp_sgd/dp_mnist:
BUILD  dp_mnist.py

differential_privacy/dp_sgd/dp_optimizer:
BUILD  dp_optimizer.py  dp_pca.py  sanitizer.py  utils.py

differential_privacy/dp_sgd/per_example_gradients:
BUILD  per_example_gradients.py

differential_privacy/privacy_accountant:
python  tf

differential_privacy/privacy_accountant/python:
BUILD  gaussian_moments.py

differential_privacy/privacy_accountant/tf:
accountant.py  accountant_test.py  BUILD

# List the data.
$ ls -R data/

./data:
mnist_test.tfrecord  mnist_train.tfrecord

# Build the codes.
$ bazel build -c opt differential_privacy/...

# Run the mnist differntial privacy training codes.
$ bazel-bin/differential_privacy/dp_sgd/dp_mnist/dp_mnist \
    --training_data_path=data/mnist_train.tfrecord \
    --eval_data_path=data/mnist_test.tfrecord \
    --save_path=/tmp/mnist_dir

...
step: 1
step: 2
...
step: 9
spent privacy: eps 0.1250 delta 0.72709
spent privacy: eps 0.2500 delta 0.24708
spent privacy: eps 0.5000 delta 0.0029139
spent privacy: eps 1.0000 delta 6.494e-10
spent privacy: eps 2.0000 delta 8.2242e-24
spent privacy: eps 4.0000 delta 1.319e-51
spent privacy: eps 8.0000 delta 3.3927e-107
train_accuracy: 0.53
eval_accuracy: 0.53
...

$ ls /tmp/mnist_dir/
checkpoint  ckpt  ckpt.meta  results-0.json
```
