#! /bin/bash
# brief: Import Fast R-CNN models
# author: Abhishek Dutta
# author: Hakan Bilen

# Models are written to <MATCONVNET>/data/models-import/fast-rcnn
# You can delete <MATCONVNET>/data/models-import/fast-rcnn/fast_rcnn_models.tgz

# TODO apply patch to prototxt which will resize the outputs of cls layers from 205 -> 1000 (maybe sed?)

overwrite=no

# urls for all FRCNN models
FRCNN_CAFFENET_PROTO_URL=https://raw.githubusercontent.com/rbgirshick/fast-rcnn/master/models/CaffeNet/test.prototxt
FRCNN_VGGM1K_PROTO_URL=https://raw.githubusercontent.com/rbgirshick/fast-rcnn/master/models/VGG_CNN_M_1024/test.prototxt
FRCNN_VGG16_PROTO_URL=https://raw.githubusercontent.com/rbgirshick/fast-rcnn/master/models/VGG16/test.prototxt

FRCNN_MODEL_URL=https://people.eecs.berkeley.edu/~rbg/fast-rcnn-data/fast_rcnn_models.tgz

# source: https://github.com/rbgirshick/fast-rcnn/blob/90e75082f087596f28173546cba615d41f0d38fe/lib/fast_rcnn/config.py
FRCNN_AVERAGE_COLOR="(122.7717, 115.9465, 102.9801)"
FRCNN_CLASSES="('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')"

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

converter="python $SCRIPTPATH/import-caffe.py"
data="$SCRIPTPATH/../data/models-import"

mkdir -p "$data/tmp/fast-rcnn"

function get()
{
    "$SCRIPTPATH/get-file.sh" "$data/tmp/fast-rcnn" "$1"
}

# --------------------------------------------------------------------
# FCN models
# --------------------------------------------------------------------

if true
then
    echo "Downloading pre-trained fast-rcnn model files (this may take some time) ..."
    get $FRCNN_CAFFENET_PROTO_URL
    mv $data/tmp/fast-rcnn/test.prototxt $data/tmp/fast-rcnn/caffenet_test.prototxt
    get $FRCNN_VGGM1K_PROTO_URL
    mv $data/tmp/fast-rcnn/test.prototxt $data/tmp/fast-rcnn/vggm1k_test.prototxt
    get $FRCNN_VGG16_PROTO_URL
    mv $data/tmp/fast-rcnn/test.prototxt $data/tmp/fast-rcnn/vgg16_test.prototxt
    get $FRCNN_MODEL_URL
    (cd $data/tmp/fast-rcnn ; tar -zxf fast_rcnn_models.tgz)
fi

if true
then

    ins=( \
        caffenet_fast_rcnn_iter_40000 \
        vgg_cnn_m_1024_fast_rcnn_iter_40000 \
        vgg16_fast_rcnn_iter_40000)
    protos=( \
        caffenet_test \
        vggm1k_test \
        vgg16_test)
    outs=( \
        fast-rcnn-caffenet-pascal07-dagnn \
        fast-rcnn-vggm1k-pascal07-dagnn \
        fast-rcnn-vgg16-pascal07-dagnn \
        fast-rcnn-vgg16-pascal07-12-dagnn \
        fast-rcnn-vgg16-pascal12-dagnn \
        )

    for ((i=0;i<${#ins[@]};++i)); do
        in="$data/tmp/fast-rcnn/fast_rcnn_models/${ins[i]}"
        out="$data/${outs[i]}.mat"
        if test -f "$out" -a -z "$overwrite"
        then
            echo "$out exists; skipping."
        else
            echo "Exporting caffe model to matconvnet format (this may also take some time) ..."
            $converter \
                --caffe-variant=caffe_fastrcnn \
                --remove-dropout \
                --remove-loss \
                --average-value="${FRCNN_AVERAGE_COLOR}" \
                --full-image-size=[600] \
                --class-names="${FRCNN_CLASSES}" \
                $data/tmp/fast-rcnn/"${protos[i]}".prototxt \
                --output-format=dagnn \
                --caffe-data="$in".caffemodel \
                "$out"
        fi
    done
fi


echo "Note: you may now delete the models-import/tmp directory"
