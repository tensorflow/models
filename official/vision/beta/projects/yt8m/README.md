# YouTube-8M Tensorflow Starter Code (tf2 version)

This repo contains starter code for training and evaluating machine learning
models over the [YouTube-8M](https://research.google.com/youtube8m/) dataset. 
This is the Tensorflow2 version of the original starter code: 
[YouTube-8M Tensorflow Starter Code](https://github.com/google/youtube-8m) which was tested on Tensorflow 1.14. (The code gives an end-to-end working example for reading the
dataset, training a TensorFlow model, and evaluating the performance of the
model). Functionalities are maintained, while necessary migrations were done to accomodate running on tf2 environment.

### Requirements

The starter code requires Tensorflow. If you haven't installed it yet, follow
the instructions on [tensorflow.org](https://www.tensorflow.org/install/). This
code has been tested with Tensorflow 2.4.0. Going forward, we will continue to
target the latest released version of Tensorflow.

Please verify that you have Python 3.6+ and Tensorflow 2.4.0 or higher installed
by running the following commands:

```sh
python --version
python -c 'import tensorflow as tf; print(tf.__version__)'
```

Refer to the [instructions here](https://github.com/tensorflow/models/tree/master/official#running-the-models)
for using the model in this repo. Make sure to add the models folder to your Python path.

#### Using GPUs

If your Tensorflow installation has GPU support (which should have been provided with  `pip
install tensorflow` for any version above Tensorflow1.15), this code will make use of all of your compatible GPUs.
You can verify your installation by running

```
tf.config.list_physical_devices('GPU')
```

This will print out something like the following for each of your compatible
GPUs.

```
I tensorflow/core/common_runtime/gpu/gpu_device.cc:1720] Found device 0 with properties:
pciBusID: 0000:00:04.0 name: Tesla P100-PCIE-16GB computeCapability: 6.0
coreClock: 1.3285GHz coreCount: 56 deviceMemorySize: 15.90GiB deviceMemoryBandwidth: 681.88GiB/s
...
```


### Train video-level model on frame-level features and inference at segment-level.

The code can be run in three different modes: `train / train_and_eval / eval`.   
Run `yt8m_train.py` and specify which mode you wish to execute. Training is done
using frame-level features, while inference is done at segment-level.


The following commands will train a model on Google Cloud over frame-level
features.

```bash
python3 yt8m_train.py --mode='train' 
     --experiment='yt8m_experiment' 
     --train_data_pattern='gs://youtube8m-ml/2/frame/train/train*.tfrecord' 
     --model_dir=$MODEL_DIR
     --train_dir=$TRAIN_DIR
     --config_file=[].yaml
     --start_new_model
```

In order to run evaluation after each training epoch, set the mode to 'train_and_eval'

```bash
python3 yt8m_train.py --mode='train_and_eval' 
     --experiment='yt8m_experiment' 
     --train_data_pattern='gs://youtube8m-ml/2/frame/train/train*.tfrecord' 
     --eval_data_pattern='gs://youtube8m-ml/3/frame/validate/validate*.tfrecord'
     --model_dir=$MODEL_DIR
     --train_dir=$TRAIN_DIR
     --config_file=[].yaml
     --start_new_model
```

Running on evaluation mode loads saved checkpoint and runs inference 

```bash
python3 yt8m_train.py --mode='eval' 
     --experiment='yt8m_experiment' 
     --eval_data_pattern='gs://youtube8m-ml/3/frame/test/test*.tfrecord' 
     --model_dir=$MODEL_DIR
     --train_dir=$TRAIN_DIR
     --config_file=[].yaml
```
#### Tensorboard

```sh

```


Once these job starts executing you will see outputs similar to the following:
```
train | step:    141 | training until step 188... 
train | step:    188 | steps/sec:    0.1 | output:       
    {'learning_rate': 0.009999976,                                         
     'model_loss': 0.009135981,   
     'total_loss': 0.009622246,                                      
     'training_loss': 0.009622246} 
```

and the following for evaluation:

```
eval | step:    188 | running 11 steps of evaluation...
eval | step:    188 | eval time:   48.1 | output:            
    {'aps': 0.0011551170885774754,                         
     'avg_hit_at_one': 0.16844223484848486,  
     'avg_perr': 0.08853457752011464,    
     'gap': 0.02865141526257211, 
     'model_loss': 0.015846167,             
     'total_loss': 0.01632894,                                            
     'validation_loss': 0.01632894}
```
