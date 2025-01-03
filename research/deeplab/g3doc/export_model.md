# Export trained deeplab model to frozen inference graph

After model training finishes, you could export it to a frozen TensorFlow
inference graph proto. Your trained model checkpoint usually includes the
following files:

*   model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001,
*   model.ckpt-${CHECKPOINT_NUMBER}.index
*   model.ckpt-${CHECKPOINT_NUMBER}.meta

After you have identified a candidate checkpoint to export, you can run the
following commandline to export to a frozen graph:

```bash
# From tensorflow/models/research/
# Assume all checkpoint files share the same path prefix `${CHECKPOINT_PATH}`.
python deeplab/export_model.py \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --export_path=${OUTPUT_DIR}/frozen_inference_graph.pb
```

Please also add other model specific flags as you use for training, such as
`model_variant`, `add_image_level_feature`, etc.
