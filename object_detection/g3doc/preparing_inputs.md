# Preparing Inputs

Tensorflow Object Detection API reads data using the TFRecord file format. Two
sample scripts (`create_pascal_tf_record.py` and `create_pet_tf_record.py`) are
provided to convert from the PASCAL VOC dataset and Oxford-IIIT Pet dataset to
TFRecords.

## Generating the PASCAL VOC TFRecord files.

The raw 2012 PASCAL VOC data set is located
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
To download, extract and convert it to TFRecords, run the following commands
below:

```bash
# From tensorflow/models
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xvf VOCtrainval_11-May-2012.tar
python object_detection/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=train \
    --output_path=pascal_train.record
python object_detection/create_pascal_tf_record.py \
    --label_map_path=object_detection/data/pascal_label_map.pbtxt \
    --data_dir=VOCdevkit --year=VOC2012 --set=val \
    --output_path=pascal_val.record
```

You should end up with two TFRecord files named `pascal_train.record` and
`pascal_val.record` in the `tensorflow/models` directory.

The label map for the PASCAL VOC data set can be found at
`object_detection/data/pascal_label_map.pbtxt`.

## Generating the Oxford-IIIT Pet TFRecord files.

The Oxford-IIIT Pet data set is located on
[their website](http://www.robots.ox.ac.uk/~vgg/data/pets/).
To download, extract and convert it to TFRecrods, run the following commands
below:

```bash
# From tensorflow/models
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xvf annotations.tar.gz
tar -xvf images.tar.gz
python object_detection/create_pet_tf_record.py \
    --label_map_path=object_detection/data/pet_label_map.pbtxt \
    --data_dir=`pwd` \
    --output_dir=`pwd`
```

You should end up with two TFRecord files named `pet_train.record` and
`pet_val.record` in the `tensorflow/models` directory.

The label map for the Pet dataset can be found at
`object_detection/data/pet_label_map.pbtxt`.
