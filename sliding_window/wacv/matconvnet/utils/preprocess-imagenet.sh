#!/bin/bash
# file: preprocess-imagenet.sh
# auhtor: Andrea Vedaldi

# Use as:
#   preprocess-imagenet.sh SRC_PATH DEST_PATH
#
# The script creates a copy of the ImageNet ILSVRC CLS-LOC challenge
# data while rescaling the images. Images are rescaled to a minimum
# side of 256 pixels. The data is supposed to be in the format defined
# by examples/cnn_imagenet_setup_data.m
#
# Note that the default scripts in MatConvNet expect the following structure:
#
#   ILSVRC2012/ILSVRC2012_devkit_t12/
#   ILSVRC2012/images/train/n01440764/
#   ILSVRC2012/images/train/n01484850/
#   ...
#   ILSVRC2012/images/train/n15075141/
#   ILSVRC2012/images/val/ILSVRC2012_val_00000001.JPEG
#   ILSVRC2012/images/val/ILSVRC2012_val_00000002.JPEG
#   ...
#   ILSVRC2012/images/val/ILSVRC2012_val_00050000.JPEG
#   ILSVRC2012/images/test/ILSVRC2012_test_00000001.JPEG
#   ILSVRC2012/images/test/ILSVRC2012_test_00000002.JPEG
#   ...
#   ILSVRC2012/images/test/ILSVRC2012_test_00100000.JPEG
#
# Symbolic links within the ILSVRC2012/images hierarchy are supported
# by this script.
#
# Example:
#    Create a copy of the ILSVRC2012 data in the data/ILSVRC2012
#    subfolder. Create a link to a ramdisk directory
#    data/ram/ILSVRC2012 to contain the transformed images (provided
#    that your server has GBs of RAM!). Then:
#
#    cd <MatConvNet>
#    ./utils/preprocess-imagenet data/ILSVRC2012 data/ram/ILSVRC2012

data=$1
ram=$2

# graphics magick (much faster)
num_cpus=1
method=gm

# image size
size=256 # most common
#size=310 # for inception

# image magick
# num_cpus=8
# method=im

mkdir -p "$ram"/images ;
rsync -rv --chmod=ugo=rwX "$data"/*devkit* "$ram/"

function convert_some_im()
{
    out="$1"
    shift
    size="$1"
    shift
    for infile in "$@"
    do
        outfile="$out/$(basename $infile)"
        if test -f "$outfile"
        then
            continue ;
        fi
        convert "${infile}" \
            -verbose \
            -quality 90 \
            -colorspace RGB \
            -resize "${size}x${size}^" \
            JPEG:"${outfile}.temp"
        mv "${outfile}.temp" "$outfile"
    done
}
export -f convert_some_im

function convert_some_gm()
{
    gm=gm
    out="$1"
    shift
    size="$1"
    shift
    for infile in "$@"
    do
        outfile="$out/$(basename $infile)"
        if test -f "$outfile"
        then
            continue ;
        fi
        echo convert "'${infile}'" \
            -verbose \
            -quality 90 \
            -colorspace RGB \
            -resize "${size}x${size}^" \
            "JPEG:'${outfile}'"
    done | ${gm} batch -echo on -feedback on -
}
export -f convert_some_gm

dirs=$(find $data/images/* -maxdepth 2 -type d)
for d in $dirs
do
    sub=${d#${data}/images/}
    out="$ram/images/$sub"
    echo "Converting $d -> $out"
    mkdir -p "$out"
    find "$d" -maxdepth 1 -type f -name '*.JPEG' | \
        xargs -n 1000 --max-procs=$num_cpus \
        bash -c "convert_some_$method \"$out\" ${size} \"\$@\" " _
done

# copy any symlink
find "$data/images/" -type l -printf '%P\n' | \
  rsync -lv --files-from=- "$data/images/" "$ram/images/"
