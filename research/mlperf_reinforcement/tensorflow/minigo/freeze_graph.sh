#!/bin/bash

python3 `python3 -c "import tensorflow as tf;print (tf.__path__[0])"`/python/tools/freeze_graph.py --input_graph=minigo.pbtxt --output_graph=$1.transformed.pb --input_binary=False --input_checkpoint=$1 --output_node_names=policy_output,value_output
