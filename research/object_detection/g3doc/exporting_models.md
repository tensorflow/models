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
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory ${EXPORT_DIR}
```

Afterwards, you should see the directory ${EXPORT_DIR} containing the following:

* output_inference_graph.pb, the frozen graph format of the exported model
* saved_model/, a directory containing the saved model format of the exported model
* model.ckpt.*, the model checkpoints used for exporting
* checkpoint, a file specifying to restore included checkpoint files
