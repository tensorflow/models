python -u ~/source/tensorflow/tensorflow/python/tools/freeze_graph.py \
  --input_graph=my_inception_v4.pb \
  --input_checkpoint=/tmp/my_train/model.ckpt-120927 \
  --output_graph=./my_inception_v4_freeze.pb \
  --input_binary=True \
  --output_node_name=InceptionV4/Logits/Predictions
