# Shakespeare character LSTM model

This is an implemention of a simple character LSTM used to generate text.

## Instructions

First download the source data:

```
wget https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt
```

Note that files other than shakepeare.txt can also be used to train the model to generater other text.

Then train the model:

```python
python3 shakespeare_main.py --training_data shakespeare.txt \
    --model_dir /tmp/shakespeare
```

This will place model checkpoints in `/tmp/shakespeare`, so that we can use them to make predictions.

Then generate predictions:

```python
python3 shakespeare_main.py --training_data shakespeare.txt \
    --model_dir /tmp/shakespeare --notrain --predict_context=ROMEO:
```

Change `--predict_context` and `--predict_length` to suit your needs.
