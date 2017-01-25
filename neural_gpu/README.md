# NeuralGPU
Code for the Neural GPU model as described
in [[http://arxiv.org/abs/1511.08228]].

Requirements:
* TensorFlow (see tensorflow.org for how to install)

The model can be trained on the following algorithmic tasks:

* `sort` - Sort a symbol list
* `kvsort` - Sort symbol keys in dictionary
* `id` - Return the same symbol list
* `rev` - Reverse a symbol list
* `rev2` - Reverse a symbol dictionary by key
* `incr` - Add one to a symbol value
* `add` - Long decimal addition
* `left` - First symbol in list
* `right` - Last symbol in list
* `left-shift` - Left shift a symbol list
* `right-shift` - Right shift a symbol list
* `bmul` - Long binary multiplication
* `mul` - Long decimal multiplication
* `dup` - Duplicate a symbol list with padding
* `badd` - Long binary addition
* `qadd` - Long quaternary addition
* `search` - Search for symbol key in dictionary

It can also be trained on the WMT English-French translation task:

* `wmt` - WMT English-French translation (data will be downloaded)

The value range for symbols are defined by the `vocab_size` flag.
In particular, the values are in the range `vocab_size - 1`.
So if you set `--vocab_size=16` (the default) then `--problem=rev`
will be reversing lists of 15 symbols, and `--problem=id` will be identity
on a list of up to 15 symbols.


To train the model on the binary multiplication task run:

```
python neural_gpu_trainer.py --problem=bmul
```

This trains the Extended Neural GPU, to train the original model run:

```
python neural_gpu_trainer.py --problem=bmul --beam_size=0
```

While training, interim / checkpoint model parameters will be
written to `/tmp/neural_gpu/`.

Once the amount of error gets down to what you're comfortable
with, hit `Ctrl-C` to stop the training process. The latest
model parameters will be in `/tmp/neural_gpu/neural_gpu.ckpt-<step>`
and used on any subsequent run.

To evaluate a trained model on how well it decodes run:

```
python neural_gpu_trainer.py --problem=bmul --mode=1
```

To interact with a model (experimental, see code) run:

```
python neural_gpu_trainer.py --problem=bmul --mode=2
```

Maintained by Lukasz Kaiser (lukaszkaiser)
