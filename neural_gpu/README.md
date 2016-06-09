# NeuralGPU
Code for the Neural GPU model as described
in [[http://arxiv.org/abs/1511.08228]].

Requirements:
* TensorFlow (see tensorflow.org for how to install)
* Matplotlib for Python (sudo apt-get install python-matplotlib)

The model can be trained on the following algorithmic tasks:

* `sort` - Sort a decimal list
* `kvsort` - Sort decimal keys in dictionary
* `id` - Return the same decimal list
* `rev` - Reverse a decimal list
* `rev2` - Reverse a decimal dictionary by key
* `incr` - Add one to a decimal
* `add` - Long decimal addition
* `left` - First decimal in list
* `right` - Last decimal in list
* `left-shift` - Left shift a decimal list
* `right-shift` - Right shift a decimal list
* `bmul` - Long binary multiplication
* `mul` - Long decimal multiplication
* `dup` - Duplicate a decimal list with padding
* `badd` - Long binary addition
* `qadd` - Long quaternary addition
* `search` - Search for decimal key in dictionary

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
