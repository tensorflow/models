#!/bin/bash

# Run script for MLPerf
python3 ncf_main.py -dataset ml-20m -hooks "" -hr_threshold 0.9562 -train_epochs 100 -learning_rate 0.0005 -batch_size 2048 -layers 256,256,128,64 -num_factors 64

