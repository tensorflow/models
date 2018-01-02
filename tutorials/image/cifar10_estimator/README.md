CIFAR-10 is a common benchmark in machine learning for image recognition.

http://www.cs.toronto.edu/~kriz/cifar.html

Code in this directory focuses on how to use TensorFlow Estimators to train and 
evaluate a CIFAR-10 ResNet model on:

* A single host with one CPU;
* A single host with multiple GPUs;
* Multiple hosts with CPU or multiple GPUs;

Before trying to run the model we highly encourage you to read all the README.

## Prerequisite

1. [Install](https://www.tensorflow.org/install/) TensorFlow version 1.2.1 or
later.

2. Download the CIFAR-10 dataset and generate TFRecord files using the provided
script.  The script and associated command below will download the CIFAR-10
dataset and then generate a TFRecord for the training, validation, and
evaluation datasets. 

```shell
python generate_cifar10_tfrecords.py --data-dir=${PWD}/cifar-10-data
```

After running the command above, you should see the following files in the
--data-dir (```ls -R cifar-10-data```):

* train.tfrecords
* validation.tfrecords
* eval.tfrecords


## Training on a single machine with GPUs or CPU

Run the training on CPU only. After training, it runs the evaluation.

```
python cifar10_main.py --data-dir=${PWD}/cifar-10-data \
                       --job-dir=/tmp/cifar10 \
                       --num-gpus=0 \
                       --train-steps=1000
```

Run the model on 2 GPUs using CPU as parameter server. After training, it runs
the evaluation.
```
python cifar10_main.py --data-dir=${PWD}/cifar-10-data \
                       --job-dir=/tmp/cifar10 \
                       --num-gpus=2 \
                       --train-steps=1000
```

Run the model on 2 GPUs using GPU as parameter server.
It will run an experiment, which for local setting basically means it will run
stop training
a couple of times to perform evaluation.

```
python cifar10_main.py --data-dir=${PWD}/cifar-10-data \
                       --job-dir=/tmp/cifar10 \
                       --variable-strategy GPU \
                       --num-gpus=2 \
```

There are more command line flags to play with; run
`python cifar10_main.py --help` for details.

## Run distributed training

### (Optional) Running on Google Cloud Machine Learning Engine

This example can be run on Google Cloud Machine Learning Engine (ML Engine),
which will configure the environment and take care of running workers,
parameters servers, and masters in a fault tolerant way.

To install the command line tool, and set up a project and billing, see the
quickstart [here](https://cloud.google.com/ml-engine/docs/quickstarts/command-line).

You'll also need a Google Cloud Storage bucket for the data. If you followed the
instructions above, you can just run:

```
MY_BUCKET=gs://<my-bucket-name>
gsutil cp -r ${PWD}/cifar-10-data $MY_BUCKET/
```

Then run the following command from the `tutorials/image` directory of this
repository (the parent directory of this README):

```
gcloud ml-engine jobs submit training cifarmultigpu \
    --runtime-version 1.2 \
    --job-dir=$MY_BUCKET/model_dirs/cifarmultigpu \
    --config cifar10_estimator/cmle_config.yaml \
    --package-path cifar10_estimator/ \
    --module-name cifar10_estimator.cifar10_main \
    -- \
    --data-dir=$MY_BUCKET/cifar-10-data \
    --num-gpus=4 \
    --train-steps=1000
```


### Set TF_CONFIG

Considering that you already have multiple hosts configured, all you need is a
`TF_CONFIG` environment variable on each host. You can set up the hosts manually
or check [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) for
instructions about how to set up a Cluster.

The `TF_CONFIG` will be used by the `RunConfig` to know the existing hosts and
their task: `master`, `ps` or `worker`.

Here's an example of `TF_CONFIG`.

```python
cluster = {'master': ['master-ip:8000'],
           'ps': ['ps-ip:8000'],
           'worker': ['worker-ip:8000']}

TF_CONFIG = json.dumps(
  {'cluster': cluster,
   'task': {'type': master, 'index': 0},
   'model_dir': 'gs://<bucket_path>/<dir_path>',
   'environment': 'cloud'
  })
```

*Cluster*

A cluster spec, which is basically a dictionary that describes all of the tasks
in the cluster. More about it [here](https://www.tensorflow.org/deploy/distributed).

In this cluster spec we are defining a cluster with 1 master, 1 ps and 1 worker.

* `ps`: saves the parameters among all workers. All workers can
   read/write/update the parameters for model via ps. As some models are
   extremely large the parameters are shared among the ps (each ps stores a
   subset).

* `worker`: does the training.

* `master`: basically a special worker, it does training, but also restores and
   saves checkpoints and do evaluation.

*Task*

The Task defines what is the role of the current node, for this example the node
is the master on index 0 on the cluster spec, the task will be different for
each node. An example of the `TF_CONFIG` for a worker would be:

```python
cluster = {'master': ['master-ip:8000'],
           'ps': ['ps-ip:8000'],
           'worker': ['worker-ip:8000']}

TF_CONFIG = json.dumps(
  {'cluster': cluster,
   'task': {'type': worker, 'index': 0},
   'model_dir': 'gs://<bucket_path>/<dir_path>',
   'environment': 'cloud'
  })
```

*Model_dir*

This is the path where the master will save the checkpoints, graph and
TensorBoard files. For a multi host environment you may want to use a
Distributed File System, Google Storage and DFS are supported.

*Environment*

By the default environment is *local*, for a distributed setting we need to
change it to *cloud*.

### Running script

Once you have a `TF_CONFIG` configured properly on each host you're ready to run
on distributed settings.

#### Master
Run this on master:
Runs an Experiment in sync mode on 4 GPUs using CPU as parameter server for
40000 steps. It will run evaluation a couple of times during training. The
num_workers arugument is used only to update the learning rate correctly. Make
sure the model_dir is the same as defined on the TF_CONFIG.

```shell
python cifar10_main.py --data-dir=gs://path/cifar-10-data \
                       --job-dir=gs://path/model_dir/ \
                       --num-gpus=4 \
                       --train-steps=40000 \
                       --sync \
                       --num-workers=2
```

*Output:*

```shell
INFO:tensorflow:Using model_dir in TF_CONFIG: gs://path/model_dir/
INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_num_ps_replicas': 1, '_keep_checkpoint_max': 5, '_task_type': u'master', '_is_chief': True, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7fd16fb2be10>, '_model_dir': 'gs://path/model_dir/', '_save_checkpoints_steps': None, '_keep_checkpoint_every_n_hours': 10000, '_session_config': intra_op_parallelism_threads: 1
gpu_options {
}
allow_soft_placement: true
, '_tf_random_seed': None, '_environment': u'cloud', '_num_worker_replicas': 1, '_task_id': 0, '_save_summary_steps': 100, '_tf_config': gpu_options {
  per_process_gpu_memory_fraction: 1.0
}
, '_evaluation_master': '', '_master': u'grpc://master-ip:8000'}
...
2017-08-01 19:59:26.496208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.09GiB
2017-08-01 19:59:26.775660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 1 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:05.0
Total memory: 11.17GiB
Free memory: 11.10GiB
...
2017-08-01 19:59:29.675171: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:316] Started server with target: grpc://localhost:8000
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_2/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_3/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_4/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_5/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_6/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/avg_pool/: (?, 16, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_5/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_6/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/avg_pool/: (?, 32, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_2/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_3/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_4/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_5/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_6/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/global_avg_pool/: (?, 64)
INFO:tensorflow:image after unit resnet/tower_0/fully_connected/: (?, 11)
INFO:tensorflow:SyncReplicasV2: replicas_to_aggregate=1; total_num_replicas=1
INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:Restoring parameters from gs://path/model_dir/model.ckpt-0
2017-08-01 19:59:37.560775: I tensorflow/core/distributed_runtime/master_session.cc:999] Start master session 156fcb55fe6648d6 with config: 
intra_op_parallelism_threads: 1
gpu_options {
  per_process_gpu_memory_fraction: 1
}
allow_soft_placement: true

INFO:tensorflow:Saving checkpoints for 1 into gs://path/model_dir/model.ckpt.
INFO:tensorflow:loss = 1.20682, step = 1
INFO:tensorflow:loss = 1.20682, learning_rate = 0.1
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_2/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_3/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_4/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_5/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_6/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/avg_pool/: (?, 16, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_5/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_6/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/avg_pool/: (?, 32, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_2/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_3/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_4/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_5/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_6/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/global_avg_pool/: (?, 64)
INFO:tensorflow:image after unit resnet/tower_0/fully_connected/: (?, 11)
INFO:tensorflow:SyncReplicasV2: replicas_to_aggregate=2; total_num_replicas=2
INFO:tensorflow:Starting evaluation at 2017-08-01-20:00:14
2017-08-01 20:00:15.745881: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:0) -> (device: 0, name: Tesla K80, pci bus id: 0000:00:04.0)
2017-08-01 20:00:15.745949: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:1) -> (device: 1, name: Tesla K80, pci bus id: 0000:00:05.0)
2017-08-01 20:00:15.745958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:2) -> (device: 2, name: Tesla K80, pci bus id: 0000:00:06.0)
2017-08-01 20:00:15.745964: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:3) -> (device: 3, name: Tesla K80, pci bus id: 0000:00:07.0)
2017-08-01 20:00:15.745969: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:4) -> (device: 4, name: Tesla K80, pci bus id: 0000:00:08.0)
2017-08-01 20:00:15.745975: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:5) -> (device: 5, name: Tesla K80, pci bus id: 0000:00:09.0)
2017-08-01 20:00:15.745987: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:6) -> (device: 6, name: Tesla K80, pci bus id: 0000:00:0a.0)
2017-08-01 20:00:15.745997: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1030] Creating TensorFlow device (/gpu:7) -> (device: 7, name: Tesla K80, pci bus id: 0000:00:0b.0)
INFO:tensorflow:Restoring parameters from gs://path/model_dir/model.ckpt-10023
INFO:tensorflow:Evaluation [1/100]
INFO:tensorflow:Evaluation [2/100]
INFO:tensorflow:Evaluation [3/100]
INFO:tensorflow:Evaluation [4/100]
INFO:tensorflow:Evaluation [5/100]
INFO:tensorflow:Evaluation [6/100]
INFO:tensorflow:Evaluation [7/100]
INFO:tensorflow:Evaluation [8/100]
INFO:tensorflow:Evaluation [9/100]
INFO:tensorflow:Evaluation [10/100]
INFO:tensorflow:Evaluation [11/100]
INFO:tensorflow:Evaluation [12/100]
INFO:tensorflow:Evaluation [13/100]
...
INFO:tensorflow:Evaluation [100/100]
INFO:tensorflow:Finished evaluation at 2017-08-01-20:00:31
INFO:tensorflow:Saving dict for global step 1: accuracy = 0.0994, global_step = 1, loss = 630.425
```

#### Worker

Run this on worker:
Runs an Experiment in sync mode on 4 GPUs using CPU as parameter server for
40000 steps. It will run evaluation a couple of times during training. Make sure
the model_dir is the same as defined on the TF_CONFIG.

```shell
python cifar10_main.py --data-dir=gs://path/cifar-10-data \
                       --job-dir=gs://path/model_dir/ \
                       --num-gpus=4 \
                       --train-steps=40000 \
                       --sync
```

*Output:*

```shell
INFO:tensorflow:Using model_dir in TF_CONFIG: gs://path/model_dir/
INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600,
'_num_ps_replicas': 1, '_keep_checkpoint_max': 5, '_task_type': u'worker',
'_is_chief': False, '_cluster_spec':
<tensorflow.python.training.server_lib.ClusterSpec object at 0x7f6918438e10>,
'_model_dir': 'gs://<path>/model_dir/',
'_save_checkpoints_steps': None, '_keep_checkpoint_every_n_hours': 10000,
'_session_config': intra_op_parallelism_threads: 1
gpu_options {
}
allow_soft_placement: true
, '_tf_random_seed': None, '_environment': u'cloud', '_num_worker_replicas': 1,
'_task_id': 0, '_save_summary_steps': 100, '_tf_config': gpu_options {
  per_process_gpu_memory_fraction: 1.0
  }
...
2017-08-01 19:59:26.496208: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 0 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:04.0
Total memory: 11.17GiB
Free memory: 11.09GiB
2017-08-01 19:59:26.775660: I tensorflow/core/common_runtime/gpu/gpu_device.cc:940] Found device 1 with properties: 
name: Tesla K80
major: 3 minor: 7 memoryClockRate (GHz) 0.8235
pciBusID 0000:00:05.0
Total memory: 11.17GiB
Free memory: 11.10GiB
...
2017-08-01 19:59:29.675171: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:316] Started server with target: grpc://localhost:8000
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_1/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_2/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_3/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_4/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_5/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage/residual_v1_6/: (?, 16, 32, 32)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/avg_pool/: (?, 16, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_1/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_2/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_3/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_4/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_5/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_1/residual_v1_6/: (?, 32, 16, 16)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/avg_pool/: (?, 32, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_1/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_2/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_3/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_4/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_5/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/stage_2/residual_v1_6/: (?, 64, 8, 8)
INFO:tensorflow:image after unit resnet/tower_0/global_avg_pool/: (?, 64)
INFO:tensorflow:image after unit resnet/tower_0/fully_connected/: (?, 11)
INFO:tensorflow:SyncReplicasV2: replicas_to_aggregate=2; total_num_replicas=2
INFO:tensorflow:Create CheckpointSaverHook.
2017-07-31 22:38:04.629150: I
tensorflow/core/distributed_runtime/master.cc:209] CreateSession still waiting
for response from worker: /job:master/replica:0/task:0
2017-07-31 22:38:09.263492: I
tensorflow/core/distributed_runtime/master_session.cc:999] Start master
session cc58f93b1e259b0c with config: 
intra_op_parallelism_threads: 1
gpu_options {
per_process_gpu_memory_fraction: 1
}
allow_soft_placement: true
INFO:tensorflow:loss = 5.82382, step = 0
INFO:tensorflow:loss = 5.82382, learning_rate = 0.8
INFO:tensorflow:Average examples/sec: 1116.92 (1116.92), step = 10
INFO:tensorflow:Average examples/sec: 1233.73 (1377.83), step = 20
INFO:tensorflow:Average examples/sec: 1485.43 (2509.3), step = 30
INFO:tensorflow:Average examples/sec: 1680.27 (2770.39), step = 40
INFO:tensorflow:Average examples/sec: 1825.38 (2788.78), step = 50
INFO:tensorflow:Average examples/sec: 1929.32 (2697.27), step = 60
INFO:tensorflow:Average examples/sec: 2015.17 (2749.05), step = 70
INFO:tensorflow:loss = 37.6272, step = 79 (19.554 sec)
INFO:tensorflow:loss = 37.6272, learning_rate = 0.8 (19.554 sec)
INFO:tensorflow:Average examples/sec: 2074.92 (2618.36), step = 80
INFO:tensorflow:Average examples/sec: 2132.71 (2744.13), step = 90
INFO:tensorflow:Average examples/sec: 2183.38 (2777.21), step = 100
INFO:tensorflow:Average examples/sec: 2224.4 (2739.03), step = 110
INFO:tensorflow:Average examples/sec: 2240.28 (2431.26), step = 120
INFO:tensorflow:Average examples/sec: 2272.12 (2739.32), step = 130
INFO:tensorflow:Average examples/sec: 2300.68 (2750.03), step = 140
INFO:tensorflow:Average examples/sec: 2325.81 (2745.63), step = 150
INFO:tensorflow:Average examples/sec: 2347.14 (2721.53), step = 160
INFO:tensorflow:Average examples/sec: 2367.74 (2754.54), step = 170
INFO:tensorflow:loss = 27.8453, step = 179 (18.893 sec)
...
```

#### PS

Run this on ps:
The ps will not do training so most of the arguments won't affect the execution

```shell
python cifar10_main.py --job-dir=gs://path/model_dir/
```

*Output:*

```shell
INFO:tensorflow:Using model_dir in TF_CONFIG: gs://path/model_dir/
INFO:tensorflow:Using config: {'_save_checkpoints_secs': 600, '_num_ps_replicas': 1, '_keep_checkpoint_max': 5, '_task_type': u'ps', '_is_chief': False, '_cluster_spec': <tensorflow.python.training.server_lib.ClusterSpec object at 0x7f48f1addf90>, '_model_dir': 'gs://path/model_dir/', '_save_checkpoints_steps': None, '_keep_checkpoint_every_n_hours': 10000, '_session_config': intra_op_parallelism_threads: 1
gpu_options {
}
allow_soft_placement: true
, '_tf_random_seed': None, '_environment': u'cloud', '_num_worker_replicas': 1, '_task_id': 0, '_save_summary_steps': 100, '_tf_config': gpu_options {
  per_process_gpu_memory_fraction: 1.0
}
, '_evaluation_master': '', '_master': u'grpc://master-ip:8000'}
2017-07-31 22:54:58.928088: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job master -> {0 -> master-ip:8000}
2017-07-31 22:54:58.928153: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job ps -> {0 -> localhost:8000}
2017-07-31 22:54:58.928160: I tensorflow/core/distributed_runtime/rpc/grpc_channel.cc:215] Initialize GrpcChannelCache for job worker -> {0 -> worker-ip:8000}
2017-07-31 22:54:58.929873: I tensorflow/core/distributed_runtime/rpc/grpc_server_lib.cc:316] Started server with target: grpc://localhost:8000
```

## Visualizing results with TensorBoard

When using Estimators you can also visualize your data in TensorBoard, with no
changes in your code. You can use TensorBoard to visualize your TensorFlow
graph, plot quantitative metrics about the execution of your graph, and show
additional data like images that pass through it.

You'll see something similar to this if you "point" TensorBoard to the
`job dir` parameter you used to train or evaluate your model.

Check TensorBoard during training or after it. Just point TensorBoard to the
model_dir you chose on the previous step.

```shell
tensorboard --log-dir="<job dir>"
```

## Warnings

When runninng `cifar10_main.py` with `--sync` argument you may see an error
similar to:

```python
File "cifar10_main.py", line 538, in <module>
    tf.app.run()
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/platform/app.py", line 48, in run
    _sys.exit(main(_sys.argv[:1] + flags_passthrough))
File "cifar10_main.py", line 518, in main
    hooks), run_config=config)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/learn_runner.py", line 210, in run
    return _execute_schedule(experiment, schedule)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/learn_runner.py", line 47, in _execute_schedule
    return task()
File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/experiment.py", line 501, in train_and_evaluate
    hooks=self._eval_hooks)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/contrib/learn/python/learn/experiment.py", line 681, in _call_evaluate
    hooks=hooks)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 292, in evaluate
    name=name)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 638, in _evaluate_model
    features, labels, model_fn_lib.ModeKeys.EVAL)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/estimator/estimator.py", line 545, in _call_model_fn
    features=features, labels=labels, **kwargs)
File "cifar10_main.py", line 331, in _resnet_model_fn
    gradvars, global_step=tf.train.get_global_step())
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/training/sync_replicas_optimizer.py", line 252, in apply_gradients
    variables.global_variables())
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py", line 170, in wrapped
    return _add_should_use_warning(fn(*args, **kwargs))
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py", line 139, in _add_should_use_warning
    wrapped = TFShouldUseWarningWrapper(x)
File "/usr/local/lib/python2.7/dist-packages/tensorflow/python/util/tf_should_use.py", line 96, in __init__
    stack = [s.strip() for s in traceback.format_stack()]
```

This should not affect your training, and should be fixed on the next releases.
