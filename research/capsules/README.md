Code for Capsule model used in the following paper:
* "Dynamic Routing between Capsules" by
Sara Sabour, Nickolas Frosst, Geoffrey E. Hinton.

Requirements:
* TensorFlow (see http://www.tensorflow.org for how to install/upgrade)
* NumPy (see http://www.numpy.org/)
* GPU

Verify if the setup is correct by running the tests, such as:
```
python layers_test.py
```

Quick MNIST test results:

* Download and extract MNIST tfrecords to $DATA_DIR/ from:
https://storage.googleapis.com/capsule_toronto/mnist_data.tar.gz
* Download and extract MNIST model checkpoint to $CKPT_DIR from:
https://storage.googleapis.com/capsule_toronto/mnist_checkpoints.tar.gz

```
python experiment.py --data_dir=$DATA_DIR/mnist_data/ --train=false \
--summary_dir=/tmp/ --checkpoint=$CKPT_DIR/mnist_checkpoint/model.ckpt-1
```

Quick CIFAR10 ensemble test results:

* Download and extract cifar10 binary version to $DATA_DIR/
  from https://www.cs.toronto.edu/~kriz/cifar.html
* Download and extract cifar10 model checkpoints to $CKPT_DIR from:
https://storage.googleapis.com/capsule_toronto/cifar_checkpoints.tar.gz
* Pass the directory that the binaries are extracted to ($DATA_DIR) as data_dir

```
python experiment.py --data_dir=$DATA_DIR --train=false --dataset=cifar10 \
--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false \
--summary_dir=/tmp/ --checkpoint=$CKPT_DIR/cifar/cifar{}/model.ckpt-600000 \
--num_trials=7
```

Sample CIFAR10 training command:

```
python experiment.py --data_dir=$DATA_DIR --dataset=cifar10 --max_steps=600000\
--hparams_override=num_prime_capsules=64,padding=SAME,leaky=true,remake=false \
--summary_dir=/tmp/
```

Sample MNIST full training command:

* To train on training-validation pass --validate=true as well.
* To train on more than one gpu pass --num_gpus=NUM_GPUS

```
python experiment.py --data_dir=$DATA_DIR/mnist_data/ --max_steps=300000\
--summary_dir=/tmp/attempt0/
```


Sample MNIST baseline training command:

```
python experiment.py --data_dir=$DATA_DIR/mnist_data/ --max_steps=300000\
--summary_dir=/tmp/attempt1/ --model=baseline
```

To test on validation during training of the above model:

Notes about running continuously during training:
* pass --validate=true during training job as well.
* It would require to have 2 gpus in total: 
one for training job one for validation job.
* If both jobs are on the same machine you would need to restrict RAM 
  consumption for each job because TensorFlow will fill all your RAM for the 
  session of your first job and your second job will fail.


```
python experiment.py --data_dir=$DATA_DIR/mnist_data/ --max_steps=300000\
--summary_dir=/tmp/attempt0/ --train=false --validate=true
```

To test/train on MultiMNIST pass --num_targets=2 and
--data_dir=$DATA_DIR/multitest_6shifted_mnist.tfrecords@10. The code to 
generate multiMNIST/MNIST records is at input_data/mnist/mnist_shift.py.

Sample code to generate multiMNIST test split:

```
python mnist_shift.py --data_dir=$DATA_DIR/mnist_data/ --split=test --shift=6 
--pad=4 --num_pairs=1000 --max_shard=100000 --multi_targets=true
```

To build expanded_mnist for affNIST generalizability pass --shift=6 --pad=6.

The code to read affNIST is to follow.

Maintained by Sara Sabour (sarasra, sasabour@google.com).
