# Namignizer

Use a variation of the [PTB](https://www.tensorflow.org/versions/r0.8/tutorials/recurrent/index.html#recurrent-neural-networks) model to recognize and generate names using the [Kaggle Baby Name Database](https://www.kaggle.com/kaggle/us-baby-names).

### API
Namignizer is implemented in Tensorflow 0.8r and uses the python package `pandas` for some data processing.

#### How to use
Download the data from Kaggle and place it in your data directory (or use the small training data provided). The example data looks like so:

```
Id,Name,Year,Gender,Count
1,Mary,1880,F,7065
2,Anna,1880,F,2604
3,Emma,1880,F,2003
4,Elizabeth,1880,F,1939
5,Minnie,1880,F,1746
6,Margaret,1880,F,1578
7,Ida,1880,F,1472
8,Alice,1880,F,1414
9,Bertha,1880,F,1320
```

But any data with the two columns: `Name` and `Count` will work.

With the data, we can then train the model:

```python
train("data/SmallNames.txt", "model/namignizer", SmallConfig)
```

And you will get the output:

```
Reading Name data in data/SmallNames.txt
Epoch: 1 Learning rate: 1.000
0.090 perplexity: 18.539 speed: 282 lps
...
0.890 perplexity: 1.478 speed: 285 lps
0.990 perplexity: 1.477 speed: 284 lps
Epoch: 13 Train Perplexity: 1.477
```

This will as a side effect write model checkpoints to the `model` directory. With this you will be able to determine the perplexity your model will give you for any arbitrary set of names like so:

```python
namignize(["mary", "ida", "gazorpazorp", "houyhnhnms", "bob"],
  tf.train.latest_checkpoint("model"), SmallConfig)
```
You will provide the same config and the same checkpoint directory. This will allow you to use a the model you just trained. You will then get a perplexity output for each name like so:

```
Name mary gives us a perplexity of 1.03105580807
Name ida gives us a perplexity of 1.07770049572
Name gazorpazorp gives us a perplexity of 175.940353394
Name houyhnhnms gives us a perplexity of 9.53870773315
Name bob gives us a perplexity of 6.03938627243
```

Finally, you will also be able generate names using the model like so:

```python
namignator(tf.train.latest_checkpoint("model"), SmallConfig)
```

Again, you will need to provide the same config and the same checkpoint directory. This will allow you to use a the model you just trained. You will then get a single generated name. Examples of output that I got when using the provided data are:

```
['b', 'e', 'r', 't', 'h', 'a', '`']
['m', 'a', 'r', 'y', '`']
['a', 'n', 'n', 'a', '`']
['m', 'a', 'r', 'y', '`']
['b', 'e', 'r', 't', 'h', 'a', '`']
['a', 'n', 'n', 'a', '`']
['e', 'l', 'i', 'z', 'a', 'b', 'e', 't', 'h', '`']
```

Notice that each name ends with a backtick. This marks the end of the name.

### Contact Info

Feel free to reach out to me at knt(at google) or k.nathaniel.tucker(at gmail)
