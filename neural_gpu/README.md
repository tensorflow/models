# NeuralGPU
Code for the Neural GPU model as described
in [[http://arxiv.org/abs/1511.08228]].

Requirements:
* TensorFlow (see tensorflow.org for how to install)
* Matplotlib for Python (sudo apt-get install python-matplotlib)

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

The value range for symbols are defined by the `niclass` and `noclass` flags.
In particular, the values are in the range `min(--niclass, noclass) - 1`.
So if you set `--niclass=33` and `--noclass=33` (the default) then `--task=rev`
will be reversing lists of 32 symbols, and `--task=id` will be identity on a
list of up to 32 symbols.


To train the model on the reverse task run:

```
python neural_gpu_trainer.py --task=rev
```

While training, interim / checkpoint model parameters will be
written to `/tmp/neural_gpu/`.

Once the amount of error gets down to what you're comfortable
with, hit `Ctrl-C` to stop the training process. The latest
model parameters will be in `/tmp/neural_gpu/neural_gpu.ckpt-<step>`
and used on any subsequent run.

To test a trained model on how well it decodes run:

```
python neural_gpu_trainer.py --task=rev --mode=1
```

To produce an animation of the result run:

```
python neural_gpu_trainer.py --task=rev --mode=1 --animate=True
```

Maintained by Lukasz Kaiser (lukaszkaiser)
