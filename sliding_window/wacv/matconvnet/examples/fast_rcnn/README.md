# Fast-RCNN demo

This folder contains an example implementation of Fast-RCNN [1] in
MatConvNet. The example trains and test on the PASCAL VOC 2007 data.

There are three entry-point scripts:

* `fast_rcnn_demo.m`: runs the original Caffe model imported in MatConvNet.
* `fast_rcnn_train.m`: trains a new model from scratch, using pre-computed proposals.
* `fast_rcnn_evaluate.m`: evaluates the trained model.

Note that the code does not ship with a proposal generation method, so
proposals must be precomputed (using e.g. edge boxes or selective
search windows).

The `fast_rcnn_demo.m` code should run out of the box, downloading the
model as needed.

To test the training code using the first GPU on your system, use
something like:

    run matlab/vl_setupnn
    addpath examples/fast_rcnn
    fast_rcnn_train('train',struct('gpus',1)) ;
    fast_rcnn_evaluate('gpu',1) ;

## References

1.  *Fast R-CNN*, R. Girshick, International Conference on Computer
    Vision (ICCV), 2015.
