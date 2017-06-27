# Preparing Inputs

Tensorflow Object Detection API reads data using the TFRecord file format. Two
sample scripts (`create_pascal_tf_record.py` and `create_pet_tf_record.py`) are
provided to convert from the PASCAL VOC dataset and Oxford-IIIT Pet dataset to
TFRecords.

## Generating the PASCAL VOC TFRecord files.

The raw 2012 PASCAL VOC data set can be downloaded
[here](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar).
Extract the tar file and run the `create_pascal_tf_record` script:

```
# From tensorflow/models/object_detection
tar -xvf VOCtrainval_11-May-2012.tar
python create_pascal_tf_record.py --data_dir=VOCdevkit \
    --year=VOC2012 --set=train --output_path=pascal_train.record
python create_pascal_tf_record.py --data_dir=/home/user/VOCdevkit \
    --year=VOC2012 --set=val --output_path=pascal_val.record
```

You should end up with two TFRecord files named pascal_train.record and
pascal_val.record in the tensorflow/models/object_detection directory.

The label map for the PASCAL VOC data set can be found at
data/pascal_label_map.pbtxt.

## Generation the Oxford-IIIT Pet TFRecord files.

The Oxford-IIIT Pet data set can be downloaded from
[their website](http://www.robots.ox.ac.uk/~vgg/data/pets/). Extract the tar
file and run the `create_pet_tf_record` script to generate TFRecords.

```
# From tensorflow/models/object_detection
tar -xvf annotations.tar.gz
tar -xvf images.tar.gz
python create_pet_tf_record.py --data_dir=`pwd` --output_dir=`pwd`
```

You should end up with two TFRecord files named pet_train.record and
pet_val.record in the tensorflow/models/object_detection directory.

The label map for the Pet dataset can be found at data/pet_label_map.pbtxt.
