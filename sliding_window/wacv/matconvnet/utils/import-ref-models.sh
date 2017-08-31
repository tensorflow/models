#! /bin/bash
# brief: Import various CNN models from the web
# author: Andrea Vedaldi

# Models are written to <MATCONVNET>/data/models
# You can delete <MATCONVNET>/data/tmp after conversion

CAFFE_URL=http://dl.caffe.berkeleyvision.org
CAFFE_GIT=https://github.com/BVLC/caffe/raw
VGG_URL=http://www.robots.ox.ac.uk/~vgg/software/deep_eval/releases
VGG_DEEPEVAL=deepeval-encoder-1.0.1
VGG_DEEPEVAL_MODELS=models-1.0.1
VGG_VERYDEEP_GIST=https://gist.githubusercontent.com/ksimonyan
VGG_VERYDEEP_URL=http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe

# Obtain the path of this script
pushd `dirname $0` > /dev/null
SCRIPTPATH=`pwd`
popd > /dev/null

converter="python $SCRIPTPATH/import-caffe.py"
data="$SCRIPTPATH/../data/models-import"

mkdir -p "$data/tmp/"{vgg,caffe}
overwrite=no

# --------------------------------------------------------------------
# VGG Very Deep
# --------------------------------------------------------------------

if true
then
    (
        # we need this for the synsets lits
        cd "$data/tmp/caffe"
        wget -c -nc $CAFFE_URL/caffe_ilsvrc12.tar.gz
        tar xzvf caffe_ilsvrc12.tar.gz
        # deep models
        cd "$data/tmp/vgg"
        wget -c -nc $VGG_VERYDEEP_GIST/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt
        wget -c -nc $VGG_VERYDEEP_GIST/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt
        wget -c -nc $VGG_VERYDEEP_URL/VGG_ILSVRC_{16,19}_layers.caffemodel
    )
fi

if true
then
    # Remark: the VD models want the `caffe` format, not the
    # `vgg-caffe` as for the Devil's models below. Preprocessing is
    # `vgg-caffe` in both cases, however.

    base="$data/tmp/vgg/"
    in=(VGG_ILSVRC_19_layers VGG_ILSVRC_16_layers)
    out=(verydeep-19 verydeep-16)
    synset=(caffe caffe)

    for ((i=0;i<${#in[@]};++i)); do
        out="$data/imagenet-vgg-${out[i]}.mat"
        if test ! -e "$out" -o "$overwrite" = yes ; then
            $converter \
                --output-format=simplenn \
                --caffe-variant=caffe \
                --preproc=vgg-caffe \
	        --remove-dropout \
                --remove-loss \
                --synsets="$data/tmp/${synset[i]}/synset_words.txt" \
                --average-value="(123.68, 116.779, 103.939)" \
                --caffe-data="$base/${in[i]}.caffemodel" \
                "$base/${in[i]}_deploy.prototxt" \
                "$out"
        else
            echo "$out exists"
        fi
    done
fi

# --------------------------------------------------------------------
# VGG Return of the Devil
# --------------------------------------------------------------------

if true
then
    (
        cd "$data/tmp/vgg"
        ln -sf "$SCRIPTPATH/proto/vgg_synset_words.txt" synset_words.txt
        wget -c -nc $VGG_URL/$VGG_DEEPEVAL.tar.gz
        tar xzvf $VGG_DEEPEVAL.tar.gz
        cd $VGG_DEEPEVAL/models
        wget -c -nc $VGG_URL/$VGG_DEEPEVAL_MODELS.tar.gz
        tar xzvf $VGG_DEEPEVAL_MODELS.tar.gz
    )
fi

if true
then
    base="$data/tmp/vgg/$VGG_DEEPEVAL/models"
    in=(CNN_F CNN_M CNN_S CNN_M_128 CNN_M_1024 CNN_M_2048)
    out=(f m s m-128 m-1024 m-2048)
    synset=(caffe vgg vgg vgg vgg vgg)

    for ((i=0;i<${#in[@]};++i)); do
        out="$data/imagenet-vgg-${out[i]}.mat"
        if test ! -e "$out" -o "$overwrite" = yes ; then
            $converter \
                --output-format=simplenn \
                --caffe-variant=vgg-caffe \
                --preproc=vgg-caffe \
	        --remove-dropout \
                --remove-loss \
                --color-format=rgb \
                --synsets="$data/tmp/${synset[i]}/synset_words.txt" \
                --average-image="$base/mean.mat" \
                --caffe-data="$base/${in[i]}/model" \
                "$base/${in[i]}/param.prototxt" \
                "$out"
        else
            echo "$out exists"
        fi
    done
fi

# --------------------------------------------------------------------
# Caffe Reference Models
# --------------------------------------------------------------------

if true
then
    (
        cd "$data/tmp/caffe"
        wget -c -nc $CAFFE_URL/caffe_reference_imagenet_model
        wget -c -nc $CAFFE_URL/caffe_alexnet_model
        wget -c -nc $CAFFE_URL/caffe_ilsvrc12.tar.gz
        wget -c -nc $CAFFE_GIT/5d0958c173ac4d4632ea4146c538a35585a3ddc4/examples/imagenet/alexnet_deploy.prototxt
        wget -c -nc $CAFFE_GIT/8198585b4a670ee2d261d436ebecbb63688da617/examples/imagenet/imagenet_deploy.prototxt
        tar xzvf caffe_ilsvrc12.tar.gz
    )
fi

if true
then
    base=$data/tmp/caffe

    out=$data/imagenet-caffe-alex.mat
    test ! -e "$out" -o "$overwrite" = yes && \
        $converter \
        --output-format=simplenn \
        --caffe-variant=caffe \
        --preproc=caffe \
	--remove-dropout \
        --remove-loss \
        --synsets="$base/synset_words.txt" \
        --average-image="$base/imagenet_mean.binaryproto" \
        --caffe-data="$base/caffe_alexnet_model" \
        "$base/alexnet_deploy.prototxt" \
        "$out"

    out=$data/imagenet-caffe-ref.mat
    test ! -e "$out" -o "$overwrite" = yes && \
        $converter \
        --output-format=simplenn \
        --caffe-variant=caffe \
        --preproc=caffe \
	--remove-dropout \
        --remove-loss \
        --synsets="$base/synset_words.txt" \
        --average-image="$base/imagenet_mean.binaryproto" \
        --caffe-data="$base/caffe_reference_imagenet_model" \
        "$base/imagenet_deploy.prototxt" \
        "$out"
fi
