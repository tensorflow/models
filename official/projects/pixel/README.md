# Language Modelling with Pixels (PIXEL)

TF2 implementation of [PIXEL](https://arxiv.org/abs/2207.06991).

### Setup

The current setup requires a numpyfied pytorch pixel model and preprocessed
data. For the pixel model, we directly convert its state_dict and saved as
numpy. For the preprocessed data, we run their pytorch implementation and save
the pixel transformed data.

Let's put these data in the directory `PATH_TO_PIXEL_DATA_DIR`, then
, to convert the numpyfied model to a tensorflow checkpoint, run

```shell
python3 utils/convert_numpy_weights_to_tf.py $PATH_TO_PIXEL_DATA_DIR
```

This will create a `pixel_encoder.ckpt`. Denote the path to this checkpoint as
`PATH_TO_PIXEL_ENCODER_CKPT`.

### Training

```shell
export PATH_TO_PIXEL_DATA_DIR=xxx
export PATH_TO_PIXEL_ENCODER_CKPT=xxx
PATH_TO_TRAINING_RECORD=$PATH_TO_PIXEL_DATA_DIR/train.tf_record-*-of-20 # path to the training record
PATH_TO_TESTING_RECORD=$PATH_TO_PIXEL_DATA_DIR/eval.tf_record # path to the evaluation record
TPU_NAME="<tpu-name>"  # The name assigned while creating a Cloud TPU
MODEL_DIR=/tmp/pixel_sst2 # directory to store the experiment
# Now launch the experiment.
python3 -m official.projects.pixel.train \
  --experiment=pixel_sst2_finetune \
  --params_override="task.train_data.input_path=${PATH_TO_TRAINING_RECORD},task.validation_data.input_path=${PATH_TO_TESTING_RECORD},runtime.distribution_strategy=tpu,init_checkpoint=$PATH_TO_PIXEL_ENCODER_CKPT"
  --mode=train_and_eval \
  --tpu=$TPU_NAME \
  --model_dir=$MODEL_DIR
```