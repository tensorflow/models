# Exporting a trained model for inference

After your model has been trained, you should export it to a Tensorflow
graph proto. A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command from tensorflow/models/research:

``` bash
# From tensorflow/models/research/
INPUT_TYPE=image_tensor
PIPELINE_CONFIG_PATH={path to pipeline config file}
TRAINED_CKPT_PREFIX={path to model.ckpt}
EXPORT_DIR={path to folder that will be used for export}
python object_detection/export_inference_graph.py \
    --input_type=${INPUT_TYPE} \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix=${TRAINED_CKPT_PREFIX} \
    --output_directory=${EXPORT_DIR}
```

NOTE: We are configuring our exported model to ingest 4-D image tensors. We can
also configure the exported model to take encoded images or serialized
`tf.Example`s.

After export, you should see the directory ${EXPORT_DIR} containing the following:

* saved_model/, a directory containing the saved model format of the exported model
* frozen_inference_graph.pb, the frozen graph format of the exported model
* model.ckpt.*, the model checkpoints used for exporting
* checkpoint, a file specifying to restore included checkpoint files
* pipeline.config, pipeline config file for the exported model
