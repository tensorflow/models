#! /bin/bash
# brief: Import various CNN models from the web
# author: Karel Lenc and Andrea Vedaldi

# Models are written to <MATCONVNET>/data/models
# You can delete <MATCONVNET>/data/tmp after conversion

# TODO apply patch to prototxt which will resize the outputs of cls layers from 205 -> 1000 (maybe sed?)

overwrite=yes

CAFFE_URL=http://dl.caffe.berkeleyvision.org/
GOOGLENET_PROTO_URL=http://vision.princeton.edu/pvt/GoogLeNet/ImageNet/train_val_googlenet.prototxt
GOOGLENET_MODEL_URL=http://vision.princeton.edu/pvt/GoogLeNet/ImageNet/imagenet_googlenet.caffemodel
GOOGLENET_MEAN_URL=http://vision.princeton.edu/pvt/GoogLeNet/ImageNet/imagenet_mean.binaryproto

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

#converter="python -m pdb $SCRIPTPATH/import-caffe.py"
converter="python $SCRIPTPATH/import-caffe.py"
data="$SCRIPTPATH/../data/models-import"

mkdir -pv "$data/tmp/googlenet"

function get()
{
    "$SCRIPTPATH/get-file.sh" "$data/tmp/googlenet" "$1"
}

# --------------------------------------------------------------------
# GoogLeNet
# --------------------------------------------------------------------

get "$CAFFE_URL/caffe_ilsvrc12.tar.gz"
(cd "$data/tmp/googlenet" ; tar xzvf caffe_ilsvrc12.tar.gz)

get "$GOOGLENET_PROTO_URL"
get "$GOOGLENET_MODEL_URL"
get "$GOOGLENET_MEAN_URL"

(
    cd "$data/tmp/googlenet" ;
    cp -v train_val_googlenet.prototxt train_val_googlenet_patched.prototxt
    patch -Np0 < "$SCRIPTPATH/proto/googlenet_prototxt_patch.diff"
)

base="$data/tmp/googlenet"
out="$data/imagenet-googlenet-dag.mat"

if test -f "$out" -a -z "$overwrite"
then
    echo "$out exists; skipping."
else
    $converter \
        --caffe-variant=caffe_0115 \
        --preproc=vgg-caffe \
	--remove-dropout \
        --remove-loss \
        --append-softmax="cls3_fc" \
        --average-image="$base/imagenet_mean.binaryproto" \
        --synsets="$base/synset_words.txt" \
        --caffe-data="$base/imagenet_googlenet.caffemodel" \
        "$base/train_val_googlenet_patched.prototxt" \
        "$out"
fi
