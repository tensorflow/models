# VGSL Specs - rapid prototyping of mixed conv/LSTM networks for images.

Variable-size Graph Specification Language (VGSL) enables the specification of a
Tensor Flow graph, composed of convolutions and LSTMs, that can process
variable-sized images, from a very short definition string.

## Applications: What is VGSL Specs good for?

VGSL Specs are designed specifically to create TF graphs for:

*   Variable size images as the input. (In one or BOTH dimensions!)
*   Output an image (heat map), sequence (like text), or a category.
*   Convolutions and LSTMs are the main computing component.
*   Fixed-size images are OK too!

But wait, aren't there other systems that simplify generating TF graphs? There
are indeed, but something they all have in common is that they are designed for
fixed size images only. If you want to solve a real OCR problem, you either have
to cut the image into arbitrary sized pieces and try to stitch the results back
together, or use VGSL.

## Basic Usage

A full model, including input and the output layers, can be built using
vgsl_model.py. Alternatively you can supply your own tensors and add your own
loss function layer if you wish, using vgslspecs.py directly.

### Building a full model

Provided your problem matches the one addressed by vgsl_model, you are good to
go.

Targeted problems:

*   Images for input, either 8 bit greyscale or 24 bit color.
*   Output is 0-d (A category, like cat, dog, train, car.)
*   Output is 1-d, with either variable length or a fixed length sequence, eg
    OCR, transcription problems in general.

Currently only softmax (1 of n) outputs are supported, but it would not be
difficult to extend to logistic.

Use vgsl_train.py to train your model, and vgsl_eval.py to evaluate it. They
just call Train and Eval in vgsl_model.py.

### Model string for a full model

The model string for a full model includes the input spec, the output spec and
the layers spec in between. Example:

```
'1,0,0,3[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c105'
```

The first 4 numbers specify the standard TF tensor dimensions: [batch, height,
width, depth], except that height and/or width may be zero, allowing them to be
variable. Batch is specific only to training, and may be a different value at
recognition/inference time. Depth needs to be 1 for greyscale and 3 for color.

The model string in square brackets [] is the main model definition, which is
described [below.](#basic-layers-syntax) The output specification takes the
form:

```
O(2|1|0)(l|s|c)n output layer with n classes.
  2 (heatmap) Output is a 2-d vector map of the input (possibly at
    different scale). (Not yet supported.)
  1 (sequence) Output is a 1-d sequence of vector values.
  0 (category) Output is a 0-d single vector value.
  l uses a logistic non-linearity on the output, allowing multiple
    hot elements in any output vector value. (Not yet supported.)
  s uses a softmax non-linearity, with one-hot output in each value.
  c uses a softmax with CTC. Can only be used with s (sequence).
  NOTE Only O0s, O1s and O1c are currently supported.
```

The number of classes must match the encoding of the TF Example data set.

### Layers only - providing your own input and loss layers

You don't have to use the canned input/output modules, if you provide your
separate code to read TF Example and loss functions. First prepare your inputs:

*   A TF-conventional batch of: `images = tf.float32[batch, height, width,
    depth]`
*   A tensor of the width of each image in the batch: `widths = tf.int64[batch]`
*   A tensor of the height of each image in the batch: `heights =
    tf.int64[batch]`

Note that these can be created from individual images using
`tf.train.batch_join` with `dynamic_pad=True.`

```python
import vgslspecs
...
spec = '[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]'
vgsl = vgslspecs.VGSLSpecs(widths, heights, is_training=True)
last_layer = vgsl.Build(images, spec)
...
AddSomeLossFunction(last_layer)....
```

With some appropriate training data, this would create a world-class OCR engine!

## Basic Layers Syntax

NOTE that *all* ops input and output the standard TF convention of a 4-d tensor:
`[batch, height, width, depth]` *regardless of any collapsing of dimensions.*
This greatly simplifies things, and allows the VGSLSpecs class to track changes
to the values of widths and heights, so they can be correctly passed in to LSTM
operations, and used by any downstream CTC operation.

NOTE: in the descriptions below, `<d>` is a numeric value, and literals are
described using regular expression syntax.

NOTE: Whitespace is allowed between ops.

### Naming

Each op gets a unique name by default, based on its spec string plus its
character position in the overall specification. All the Ops take an optional
name argument in braces after the mnemonic code, but before any numeric
arguments.

### Functional ops

```
C(s|t|r|l|m)[{name}]<y>,<x>,<d> Convolves using a y,x window, with no shrinkage,
  SAME infill, d outputs, with s|t|r|l|m non-linear layer.
F(s|t|r|l|m)[{name}]<d> Fully-connected with s|t|r|l|m non-linearity and d
  outputs. Reduces height, width to 1. Input height and width must be constant.
L(f|r|b)(x|y)[s][{name}]<n> LSTM cell with n outputs.
  The LSTM must have one of:
    f runs the LSTM forward only.
    r runs the LSTM reversed only.
    b runs the LSTM bidirectionally.
  It will operate on either the x- or y-dimension, treating the other dimension
  independently (as if part of the batch).
  (Full 2-d and grid are not yet supported).
  s (optional) summarizes the output in the requested dimension,
     outputting only the final step, collapsing the dimension to a
     single element.
Do[{name}] Insert a dropout layer.
```

In the above, `(s|t|r|l|m)` specifies the type of the non-linearity:

```python
s = sigmoid
t = tanh
r = relu
l = linear (i.e., None)
m = softmax
```

Examples:

`Cr5,5,32` Runs a 5x5 Relu convolution with 32 depth/number of filters.

`Lfx{MyLSTM}128` runs a forward-only LSTM, named 'MyLSTM' in the x-dimension
with 128 outputs, treating the y dimension independently.

`Lfys64` runs a forward-only LSTM in the y-dimension with 64 outputs, treating
the x-dimension independently and collapses the y-dimension to 1 element.

### Plumbing ops

The plumbing ops allow the construction of arbitrarily complex graphs. Something
currently missing is the ability to define macros for generating say an
inception unit in multiple places.

```
[...] Execute ... networks in series (layers).
(...) Execute ... networks in parallel, with their output concatenated in depth.
S[{name}]<d>(<a>x<b>)<e>,<f> Splits one dimension, moves one part to another
  dimension.
Mp[{name}]<y>,<x> Maxpool the input, reducing the (y,x) rectangle to a single
  value.
```

In the `S` op, `<a>, <b>, <d>, <e>, <f>` are numbers.

`S` is a generalized reshape. It splits input dimension `d` into `a` x `b`,
sending the high/most significant part `a` to the high/most significant side of
dimension `e`, and the low part `b` to the high side of dimension `f`.
Exception: if `d=e=f`, then then dimension `d` is internally transposed to
`bxa`. *At least one* of `e`, `f` must be equal to `d`, so no dimension can be
totally destroyed. Either `a` or `b` can be zero, meaning whatever is left after
taking out the other, allowing dimensions to be of variable size.

NOTE: Remember the standard TF convention of a 4-d tensor: `[batch, height,
width, depth]`, so `batch=0, height=1, width=2, depth=3.`

Eg. `S3(3x50)2,3` will split the 150-element depth into 3x50, with the 3 going
to the most significant part of the width, and the 50 part staying in depth.
This will rearrange a 3x50 output parallel operation to spread the 3 output sets
over width.

### Full Examples

Example 1: A graph capable of high quality OCR.

`1,0,0,1[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]O1c105`

As layer descriptions: (Input layer is at the bottom, output at the top.)

```
O1c105: Output layer produces 1-d (sequence) output, trained with CTC,
  outputting 105 classes.
Lfx256: Forward-only LSTM in x with 256 outputs
Lrx128: Reverse-only LSTM in x with 128 outputs
Lfx128: Forward-only LSTM in x with 128 outputs
Lfys64: Dimension-summarizing LSTM, summarizing the y-dimension with 64 outputs
Mp3,3: 3 x 3 Maxpool
Ct5,5,16: 5 x 5 Convolution with 16 outputs and tanh non-linearity
[]: The body of the graph is alway expressed as a series of layers.
1,0,0,1: Input is a batch of 1 image of variable size in greyscale
```

Example 2: The STREET network for reading French street name signs end-to-end.
For a detailed description see the [FSNS dataset
paper](http://link.springer.com/chapter/10.1007%2F978-3-319-46604-0_30)

```
1,600,150,3[S2(4x150)0,2 Ct5,5,16 Mp2,2 Ct5,5,64 Mp3,3
  ([Lrys64 Lbx128][Lbys64 Lbx128][Lfys64 Lbx128]) S3(3x0)2,3
  Lfx128 Lrx128 S0(1x4)0,3 Lfx256]O1c134
```

Since networks are usually illustrated with the input at the bottom, the input
layer is at the bottom, output at the top, with 'headings' *below* the section
they introduce.

```
O1c134: Output is a 1-d sequence, trained with CTC and 134 output softmax.
Lfx256: Forward-only LSTM with 256 outputs
S0(1x4)0,3: Reshape transferring the batch of 4 tiles to the depth dimension.
Lrx128: Reverse-only LSTM with 128 outputs
Lfx128: Forward-only LSTM with 128 outputs
(Final section above)
S3(3x0)2,3: Split the outputs of the 3 parallel summarizers and spread over the
  x-dimension
  [Lfys64 Lbx128]: Summarizing LSTM downwards on the y-dimension with 64
    outputs, followed by a bi-directional LSTM in the x-dimension with 128
    outputs
  [Lbys64 Lbx128]: Summarizing bi-directional LSTM on the y-dimension with
    64 outputs, followed by a bi-directional LSTM in the x-dimension with 128
    outputs
  [Lrys64 Lbx128]: Summarizing LSTM upwards on the y-dimension with 64 outputs,
    followed by a bi-directional LSTM in the x-dimension with 128 outputs
(): In parallel (re-using the inputs and concatenating the outputs):
(Summarizing section above)
Mp3,3: 3 x 3 Maxpool
Ct5,5,64: 5 x 5 Convolution with 64 outputs and tanh non-linearity
Mp2,2: 2 x 2 Maxpool
Ct5,5,16: 5 x 5 Convolution with 16 outputs and tanh non-linearity
S2(4x150)0,2: Split the x-dimension into 4x150, converting each tiled 600x150
image into a batch of 4 150x150 images
(Convolutional input section above)
[]: The body of the graph is alway expressed as a series of layers.
1,150,600,3: Input is a batch of 1, 600x150 image in 24 bit color
```

## Variable size Tensors Under the Hood

Here are some notes about handling variable-sized images since they require some
consideration and a little bit of knowledge about what goes on inside.

A variable-sized image is an input for which the width and/or height are not
known at graph-building time, so the tensor shape contains unknown/None/-1
sizes.

Many standard NN layers, such as convolutions, are designed to cope naturally
with variable-sized images in TF and produce a variable sized image as the
output. For other layers, such as 'Fully connected' variable size is
fundamentally difficult, if not impossible to deal with, since by definition,
*all* its inputs are connected via a weight to an output. The number of inputs
therefore must be fixed.

It is possible to handle variable sized images by using sparse tensors. Some
implementations make a single variable dimension a list instead of part of the
tensor. Both these solutions suffer from completely segregating the world of
variable size from the world of fixed size, making models and their descriptions
completely non-interchangeable.

In VGSL, we use a standard 4-d Tensor, `[batch, height, width, depth]` and
either use a batch size of 1 or put up with padding of the input images to the
largest size of any element of the batch. The other price paid for this
standardization is that the user must supply a pair of tensors of shape [batch]
specifying the width and height of each input in a batch. This allows the LSTMs
in the graph to know how many iterations to execute and how to correctly
back-propagate the gradients.

The standard TF implementation of CTC also requires a tensor giving the sequence
lengths of its inputs. If the output of VGSL is going into CTC, the lengths can
be obtained using:

```python
import vgslspecs
...
spec = '[Ct5,5,16 Mp3,3 Lfys64 Lfx128 Lrx128 Lfx256]'
vgsl = vgslspecs.VGSLSpecs(widths, heights, is_training=True)
last_layer = vgsl.Build(images, spec)
seq_lengths = vgsl.GetLengths()
```

The above will provide the widths that were given in the constructor, scaled
down by the max-pool operator. The heights may be obtained using
`vgsl.GetLengths(1)`, specifying the index of the y-dimension.

NOTE that currently the only way of collapsing a dimension of unknown size to
known size (1) is through the use of a summarizing LSTM. A single summarizing
LSTM will collapse one dimension (x or y), leaving a 1-d sequence. The 1-d
sequence can then be collapsed in the other dimension to make a 0-d categorical
(softmax) or embedding (logistic) output.

Using the (parallel) op it is entirely possible to run multiple [series] of ops
that collapse x first in one and y first in the other, reducing both eventually
to a single categorical value! For eample, the following description may do
something useful with ImageNet-like problems:

```python
[Cr5,5,16 Mp2,2 Cr5,5,64 Mp3,3 ([Lfxs64 Lfys256] [Lfys64 Lfxs256]) Fr512 Fr512]
```
