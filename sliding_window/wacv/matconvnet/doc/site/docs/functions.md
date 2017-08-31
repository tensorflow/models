# Function index

MatConvNet includes several MATLAB functions organized as follows:

- [Building blocks](#core). These functions implement the CNN
  computational blocks that can be combined either manually or using
  one of the provided wrappers to construct CNNs.
- [SimpleCNN wrapper](#simplenn). SimpleNN is a lightweight wrapper
  implementing CNNs that are linear chains of computational blocks.
- [DagNN wrapper](#dagnn). DagNN is an object-oriented wrapper
  supporting more complex network topologies.
- [Other functions](#utility). These helper functions are used to
  initialize and compile MatConvNet.

There is no general training function as training depends on the
dataset and problem. Look at the `examples` subdirectory for code
showing how to train CNNs.

<a name="core"></a>

## Building blocks

- [`vl_nnbnorm`](mfiles/vl_nnbnorm.md) Batch normalization.
- [`vl_nnbilinearsampler`](mfiles/vl_nnbilinearsampler.md) Bilinear Samppler.
- [`vl_nnconv`](mfiles/vl_nnconv.md) Linear convolution by a filter.
- [`vl_nnconcat`](mfiles/vl_nnconcat.md) Concatenation.
- [`vl_nnconvt`](mfiles/vl_nnconvt.md) Convolution transpose.
- [`vl_nncrop`](mfiles/vl_nncrop.md) Cropping.
- [`vl_nndropout`](mfiles/vl_nndropout.md) Dropout.
- [`vl_nnloss`](mfiles/vl_nnloss.md) Classification log-loss.
- [`vl_nnnoffset`](mfiles/vl_nnnoffset.md) Norm-dependent offset.
- [`vl_nnnormalize`](mfiles/vl_nnnormalize.md) Local Response Normalization (LRN).
- [`vl_nnpdist`](mfiles/vl_nnpdist.md) Pairwise distances.
- [`vl_nnpool`](mfiles/vl_nnpool.md) Max and sum pooling.
- [`vl_nnrelu`](mfiles/vl_nnrelu.md) Rectified Linear Unit.
- [`vl_nnroipool`](mfiles/vl_nnroipool.md) Reegion of interest pooling.
- [`vl_nnsigmoid`](mfiles/vl_nnsigmoid.md) Sigmoid.
- [`vl_nnsoftmax`](mfiles/vl_nnsoftmax.md) Channel soft-max.
- [`vl_nnsoftmaxloss`](mfiles/vl_nnsoftmaxloss.md) *Deprecated*
- [`vl_nnspnorm`](mfiles/vl_nnspnorm.md) Spatial normalization.

<a name="simplenn"></a>

## SimpleCNN wrapper

- [`vl_simplenn`](mfiles/simplenn/vl_simplenn.md) A lightweight wrapper for
  CNNs with a linear topology.
- [`vl_simplenn_tidy`](mfiles/simplenn/vl_simplenn_tidy.md) Upgrade or
  otherwise fix a CNN.
- [`vi_simplenn_display`](mfiles/simplenn/vl_simplenn_display.md) Print
  information about the CNN architecture.
- [`vl_simplenn_move`](mfiles/simplenn/vl_simplenn_move.md) Move the CNN
  between CPU and GPU.

<a name="dagnn"></a>

## DagNN wrapper

- [`DagNN`](mfiles/+dagnn/@DagNN/DagNN.md) An object-oriented wrapper
  for CNN with complex topologies

<a name="utility"></a>

## Other functions

- [`vl_argparse`](mfiles/vl_argparse.md) A helper function to parse
  optional arguments.
- [`vl_compilenn`](mfiles/vl_compilenn.md) Compile the MEX fields in the toolbox.
- [`vl_rootnn`](mfiles/vl_rootnn.md) Return the path to the MatConvNet toolbox installation.
- [`vl_setpunn`](mfiles/vl_setupnn.md) Setup MatConvNet for use in MATLAB.
- [`vl_imreadjpeg`](mfiles/vl_imreadjpeg.md) Quickly load a batch of JPEG images.
- [`vl_taccum`](mfiles/vl_taccum.md) Accumulate tensors operating in-place when possible.
- [`vl_tmove`](mfiles/vl_tmove.md) Exchange tensors between MATLAB processes and GPUs.
- [`vl_tshow`](mfiles/vl_tshow.md) Show a tensor on screen.
