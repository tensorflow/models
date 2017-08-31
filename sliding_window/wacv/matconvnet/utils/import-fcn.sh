#! /bin/bash
# brief: Import FCN models from Caffe Model Zoo
# author: Karel Lenc and Andrea Vedaldi

# Models are written to <MATCONVNET>/data/models
# You can delete <MATCONVNET>/data/tmp after conversion

# TODO apply patch to prototxt which will resize the outputs of cls layers from 205 -> 1000 (maybe sed?)

overwrite=yes

FCN32S_PROTO_URL=https://gist.githubusercontent.com/longjon/ac410cad48a088710872/raw/fe76e342641ddb0defad95f6dc670ccc99c35a1f/fcn-32s-pascal-deploy.prototxt
FCN16S_PROTO_URL=https://gist.githubusercontent.com/longjon/d24098e083bec05e456e/raw/dd455b2978b2943a51c37ec047a0f46121d18b56/fcn-16s-pascal-deploy.prototxt
FCN8S_PROTO_URL=https://gist.githubusercontent.com/longjon/1bf3aa1e0b8e788d7e1d/raw/2711bb261ee4404faf2ddf5b9d0d2385ff3bcc3e/fcn-8s-pascal-deploy.prototxt
FCNALEX_PROTO_URL=https://gist.githubusercontent.com/shelhamer/3f2c75f3c8c71357f24c/raw/ccd0d97662e03b83e62f26bf9d870209f20f3efc/train_val.prototxt

FCN32S_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-32s-pascal.caffemodel
FCN16S_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-16s-pascal.caffemodel
FCN8S_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-8s-pascal.caffemodel
FCNALEX_MODEL_URL=http://dl.caffe.berkeleyvision.org/fcn-alexnet-pascal.caffemodel

FCN_AVERAGE_COLOR="(122.67891434, 116.66876762, 104.00698793)"

FCN_CLASSES="('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')"

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

# Use the python implementation of protocol buffers to load gigantic caffe models.
# The CPP implementation may not load these files on all systems out-of-the-box.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
converter="python $SCRIPTPATH/import-caffe.py"
data="$SCRIPTPATH/../data/models-import"

mkdir -p "$data/tmp/fcn"

if [ ! -d "$data/models/" ]; then
    mkdir "$data/models"
fi

function get()
{
    "$SCRIPTPATH/get-file.sh" "$data/tmp/fcn" "$1"
}

# --------------------------------------------------------------------
# FCN models
# --------------------------------------------------------------------

get $FCN32S_MODEL_URL
get $FCN32S_PROTO_URL
get $FCN16S_MODEL_URL
get $FCN16S_PROTO_URL
get $FCN8S_MODEL_URL
get $FCN8S_PROTO_URL

if true
then
    ins=(fcn-32s-pascal fcn-16s-pascal fcn-8s-pascal)
    outs=(pascal-fcn32s-dag pascal-fcn16s-dag pascal-fcn8s-dag)

    for ((i=0;i<${#ins[@]};++i)); do
        in="$data/tmp/fcn/${ins[i]}"
        out="$data/${outs[i]}.mat"
        if test -f "$out" -a -z "$overwrite"
        then
            echo "$out exists; skipping."
        else
            #PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp \
            $converter \
                --caffe-variant=caffe_6e3916 \
                --full-image-size=[500] \
		--remove-dropout \
                --remove-loss \
                --average-value="${FCN_AVERAGE_COLOR}" \
                --class-names="${FCN_CLASSES}" \
                --caffe-data="$in".caffemodel \
                "$in"-deploy.prototxt \
                "$out"
        fi
    done
fi
