# Object Detection Models on TensorFlow 2.0

**Note**: The repo is still under construction. More features and instructions
will be added soon.

## Prerequsite
To get started, make sure to use Tensorflow 2.1+ on Google Cloud. Also here are
a few package you need to install to get started:

```bash
sudo apt-get install -y python-tk && \
pip install Cython matplotlib opencv-python-headless pyyaml Pillow && \
pip install 'git+https://github.com/cocodataset/cocoapi#egg=pycocotools&subdirectory=PythonAPI'
```

Next, download the code from TensorFlow models github repository or use the
pre-installed Google Cloud VM.

```bash
git clone https://github.com/tensorflow/models.git
```

## Train RetinaNet on TPU
### Train a vanilla ResNet-50 based RetinaNet.

```bash
TPU_NAME="<your GCP TPU name>"
MODEL_DIR="<path to the directory to store model files>"
RESNET_CHECKPOINT="<path to the pre-trained Resnet-50 checkpoint>"
TRAIN_FILE_PATTERN="<path to the TFRecord training data>"
EVAL_FILE_PATTERN="<path to the TFRecord validation data>"
VAL_JSON_FILE="<path to the validation annotation JSON file>"
python ~/models/official/vision/detection/main.py \
  --strategy_type=tpu \
  --tpu="${TPU_NAME?}" \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --params_override="{ type: retinanet, train: { checkpoint: { path: ${RESNET_CHECKPOINT?}, prefix: resnet50/ }, train_file_pattern: ${TRAIN_FILE_PATTERN?} }, eval: { val_json_file: ${VAL_JSON_FILE?}, eval_file_pattern: ${EVAL_FILE_PATTERN?} } }"
```

### Train a custom RetinaNet using the config file.

First, create a YAML config file, e.g. *my_retinanet.yaml*. This file specifies
the parameters to be overridden, which should at least include the following
fields.

```YAML
# my_retinanet.yaml
type: 'retinanet'
train:
  train_file_pattern: <path to the TFRecord training data>
eval:
  eval_file_pattern: <path to the TFRecord validation data>
  val_json_file: <path to the validation annotation JSON file>
```

Once the YAML config file is created, you can launch the training using the
following command.

```bash
TPU_NAME="<your GCP TPU name>"
MODEL_DIR="<path to the directory to store model files>"
python ~/models/official/vision/detection/main.py \
  --strategy_type=tpu \
  --tpu="${TPU_NAME?}" \
  --model_dir="${MODEL_DIR?}" \
  --mode=train \
  --config_file="my_retinanet.yaml"
```

## Train RetinaNet on GPU

Note: Instructions are comming soon.

## References

1.  [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. IEEE
    International Conference on Computer Vision (ICCV), 2017.
