CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory focuses on how to use TensorFlow Estimators to train and evaluate a CIFAR-10 ResNet model on a single host with one CPU and potentially multiple GPUs.

<b>Prerequisite:</b>

1. Install TensorFlow version 1.2.1 or later with GPU support.

2. Download the CIFAR-10 dataset.

```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzf cifar-10-binary.tar.gz
```

<b>How to run:</b>

```shell
# After running the above commands, you should see the following in the folder
# where the data is downloaded.
$ ls -R cifar-10-batches-bin

cifar-10-batches-bin:
batches.meta.txt  data_batch_1.bin  data_batch_2.bin  data_batch_3.bin
data_batch_4.bin  data_batch_5.bin  readme.html  test_batch.bin

# Run the model on CPU only. After training, it runs the evaluation.
$ python cifar10_main.py --data_dir=/prefix/to/downloaded/data/cifar-10-batches-bin \
                         --model_dir=/tmp/cifar10 \
                         --is_cpu_ps=True \
                         --num_gpus=0 \
                         --train_steps=1000

# Run the model on CPU and 2 CPUs. After training, it runs the evaluation.
$ python cifar10_main.py --data_dir=/prefix/to/downloaded/data/cifar-10-batches-bin \
                         --model_dir=/tmp/cifar10 \
                         --is_cpu_ps=False \
                         --force_gpu_compatible=True \
                         --num_gpus=2 \
                         --train_steps=1000

# There are more command line flags to play with; check cifar10_main.py for details.
```
