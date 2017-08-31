#! /bin/bash
# brief: Import various CNN models from the web
# author: Karel Lenc and Andrea Vedaldi

# Models are written to <MATCONVNET>/data/models
# You can delete <MATCONVNET>/data/tmp after conversion

# TODO apply patch to prototxt which will resize the outputs of cls layers from 205 -> 1000 (maybe sed?)

overwrite=yes

CAFFE_URL=http://dl.caffe.berkeleyvision.org/
RESNET_URL=http://research.microsoft.com/en-us/um/people/kahe/resnet/models.zip

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

converter="python $SCRIPTPATH/import-caffe.py"
data="$SCRIPTPATH/../data/models-import"

mkdir -pv "$data/tmp/resnet"

function get()
{
    "$SCRIPTPATH/get-file.sh" "$data/tmp/resnet" "$1"
}

# --------------------------------------------------------------------
# Resnet
# --------------------------------------------------------------------

get "$CAFFE_URL/caffe_ilsvrc12.tar.gz"
(cd "$data/tmp/resnet" ; tar xzvf caffe_ilsvrc12.tar.gz)

get "$RESNET_URL"
(cd "$data/tmp/resnet" ; unzip -n models.zip)

for t in 50 101 152
do
    base="$data/tmp/resnet"
    out="$data/imagenet-resnet-$t-dag.mat"
    cdata=--caffe-data="$base/ResNet-$t-model.caffemodel"

    if test -f "$out" -a -z "$overwrite"
    then
        echo "$out exists; skipping."
    else
        $converter \
            --caffe-variant=caffe_b590f1d \
            --preproc=vgg-caffe \
	    --remove-dropout \
            --remove-loss \
            --average-image="$base/ResNet_mean.binaryproto" \
            --synsets="$base/synset_words.txt" \
            $cdata \
            "$base/ResNet-$t-deploy.prototxt" \
            "$out"
    fi
done
