# Running on Google Cloud ML Engine

The Tensorflow Object Detection API supports distributed training on Google
Cloud ML Engine. This section documents instructions on how to train and
evaluate your model using Cloud ML. The reader should complete the following
prerequistes:

1. The reader has created and configured a project on Google Cloud Platform.
See [the Cloud ML quick start guide](https://cloud.google.com/ml-engine/docs/quickstarts/command-line).
2. The reader has installed the Tensorflow Object Detection API as documented
in the [installation instructions](installation.md).
3. The reader has a valid data set and stored it in a Google Cloud Storage
bucket. See [this page](preparing_inputs.md) for instructions on how to generate
a dataset for the PASCAL VOC challenge or the Oxford-IIIT Pet dataset.
4. The reader has configured a valid Object Detection pipeline, and stored it
in a Google Cloud Storage bucket. See [this page](configuring_jobs.md) for
details on how to write a pipeline configuration.

Additionally, it is recommended users test their job by running training and
evaluation jobs for a few iterations
[locally on their own machines](running_locally.md).

## Packaging

In order to run the Tensorflow Object Detection API on Cloud ML, it must be
packaged (along with it's TF-Slim dependency and the
[pycocotools](https://github.com/cocodataset/cocoapi/tree/master/PythonAPI/pycocotools)
library). The required packages can be created with the following command

``` bash
# From tensorflow/models/research/
bash object_detection/dataset_tools/create_pycocotools_package.sh /tmp/pycocotools
python setup.py sdist
(cd slim && python setup.py sdist)
```

This will create python packages dist/object_detection-0.1.tar.gz,
slim/dist/slim-0.1.tar.gz, and /tmp/pycocotools/pycocotools-2.0.tar.gz.

## Running a Multiworker (GPU) Training Job on CMLE

Google Cloud ML requires a YAML configuration file for a multiworker training
job using GPUs. A sample YAML file is given below:

```
trainingInput:
  runtimeVersion: "1.12"
  scaleTier: CUSTOM
  masterType: standard_gpu
  workerCount: 9
  workerType: standard_gpu
  parameterServerCount: 3
  parameterServerType: standard


```

Please keep the following guidelines in mind when writing the YAML
configuration:

* A job with n workers will have n + 1 training machines (n workers + 1 master).
* The number of parameters servers used should be an odd number to prevent
  a parameter server from storing only weight variables or only bias variables
  (due to round robin parameter scheduling).
* The learning rate in the training config should be decreased when using a
  larger number of workers. Some experimentation is required to find the
  optimal learning rate.

The YAML file should be saved on the local machine (not on GCP). Once it has
been written, a user can start a training job on Cloud ML Engine using the
following command:

```bash
# From tensorflow/models/research/
gcloud ml-engine jobs submit training object_detection_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir=gs://${MODEL_DIR} \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-central1 \
    --config ${PATH_TO_LOCAL_YAML_FILE} \
    -- \
    --model_dir=gs://${MODEL_DIR} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
```

Where `${PATH_TO_LOCAL_YAML_FILE}` is the local path to the YAML configuration,
`gs://${MODEL_DIR}` specifies the directory on Google Cloud Storage where the
training checkpoints and events will be written to and
`gs://${PIPELINE_CONFIG_PATH}` points to the pipeline configuration stored on
Google Cloud Storage.

Users can monitor the progress of their training job on the [ML Engine
Dashboard](https://console.cloud.google.com/mlengine/jobs).

Note: This sample is supported for use with 1.12 runtime version.

## Running a TPU Training Job on CMLE

Launching a training job with a TPU compatible pipeline config requires using a
similar command:

```bash
gcloud ml-engine jobs submit training `whoami`_object_detection_`date +%m_%d_%Y_%H_%M_%S` \
--job-dir=gs://${MODEL_DIR} \
--packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
--module-name object_detection.model_tpu_main \
--runtime-version 1.12 \
--scale-tier BASIC_TPU \
--region us-central1 \
-- \
--tpu_zone us-central1 \
--model_dir=gs://${MODEL_DIR} \
--pipeline_config_path=gs://${PIPELINE_CONFIG_PATH}
```

In contrast with the GPU training command, there is no need to specify a YAML
file and we point to the *object_detection.model_tpu_main* binary instead of
*object_detection.model_main*. We must also now set `scale-tier` to be
`BASIC_TPU` and provide a `tpu_zone`. Finally as before `pipeline_config_path`
points to a points to the pipeline configuration stored on Google Cloud Storage
(but is now must be a TPU compatible model).

## Running an Evaluation Job on CMLE

Note: You only need to do this when using TPU for training as it does not
interleave evaluation during training as in the case of Multiworker GPU
training.

Evaluation jobs run on a single machine, so it is not necessary to write a YAML
configuration for evaluation. Run the following command to start the evaluation
job:

```bash
gcloud ml-engine jobs submit training object_detection_eval_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.12 \
    --job-dir=gs://${MODEL_DIR} \
    --packages dist/object_detection-0.1.tar.gz,slim/dist/slim-0.1.tar.gz,/tmp/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region us-central1 \
    --scale-tier BASIC_GPU \
    -- \
    --model_dir=gs://${MODEL_DIR} \
    --pipeline_config_path=gs://${PIPELINE_CONFIG_PATH} \
    --checkpoint_dir=gs://${MODEL_DIR}
```

Where `gs://${MODEL_DIR}` points to the directory on Google Cloud Storage where
training checkpoints are saved (same as the training job), as well as
to where evaluation events will be saved on Google Cloud Storage and
`gs://${PIPELINE_CONFIG_PATH}` points to where the pipeline configuration is
stored on Google Cloud Storage.

Typically one starts an evaluation job concurrently with the training job.
Note that we do not support running evaluation on TPU, so the above command
line for launching evaluation jobs is the same whether you are training
on GPU or TPU.

## Running Tensorboard

You can run Tensorboard locally on your own machine to view progress of your
training and eval jobs on Google Cloud ML. Run the following command to start
Tensorboard:

``` bash
tensorboard --logdir=gs://${YOUR_CLOUD_BUCKET}
```

Note it may Tensorboard a few minutes to populate with results.

