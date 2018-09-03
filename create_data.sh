if [[ $# -eq 0 ]] ; then
    echo 'You must provide the modelname'
    exit 1
fi

MODELNAME=$1
NETWORK=${2:-faster_rcnn_resnet101_coco_11_06_2017}
read -p "Select your config file. Type 'simple' to use our default simple config, type 'complex' or 'supercomplex' to use our default complex config, or give the path to a specific config file. You can leave blank to use the default 'faster_rcnn_resnet101_pets.config': " CONFIG

CONFIG=${CONFIG:-research/object_detection/samples/configs/faster_rcnn_resnet101_pets.config}

if [ $CONFIG = "simple" ]
then
    CONFIG="/home/ubuntu/deep-learning/install/tensorflow/simple.config"
fi

if [ $CONFIG = "complex" ]
then
    CONFIG="/home/ubuntu/deep-learning/install/tensorflow/complex.config"
fi

if [ $CONFIG = "supercomplex" ]
then
    CONFIG="/home/ubuntu/deep-learning/install/tensorflow/supercomplex.config"
fi

echo "The config file used is '$CONFIG'"

NUMBER_OF_CLASSES=$(grep -cve '^\s*$' "classes_$MODELNAME.txt")
FOLDERDATA=$MODELNAME"_data"
FOLDERRESULT=$MODELNAME"_result"
echo -e "\033[1mFolder\033[0m: $FOLDERDATA"
echo -e "\033[1mNumber of classes\033[0m: $NUMBER_OF_CLASSES"
echo ""
python create_labelmap.py --model $MODELNAME
python create_VOC_tf_record.py --source ~/models-tf/$MODELNAME --destination  ~/models-tf/$FOLDERDATA --label_map labelmap_$MODELNAME.pbtxt --train_proportion 0.9
mv labelmap_$MODELNAME.pbtxt $FOLDERDATA/.
#wget https://storage.googleapis.com/download.tensorflow.org/models/object_detection/faster_rcnn_resnet101_coco_11_06_2017.tar.gz
#tar -xvf faster_rcnn_resnet101_coco_11_06_2017.tar.gz
cp $NETWORK/model.ckpt.* $FOLDERDATA
cp $CONFIG $FOLDERDATA/network.config
sed -i "s|PATH_TO_BE_CONFIGURED|"/home/ubuntu/models-tf/$FOLDERDATA"|g" $FOLDERDATA/network.config
sed -i "s|pet_val.record|"val.record"|g" $FOLDERDATA/network.config
sed -i "s|pet_train.record|"train.record"|g" $FOLDERDATA/network.config
sed -i "s|pet_label_map.pbtxt|"labelmap_$MODELNAME.pbtxt"|g" $FOLDERDATA/network.config
sed -i "s/num_classes: 37/num_classes: $NUMBER_OF_CLASSES/" $FOLDERDATA/network.config
echo ""
echo "---------------------------"
echo ""
echo -e "\e[32mWooot!!! You should see 7 files in ./$FOLDERDATA"
echo -e "\e[97m"

mkdir $FOLDERRESULT

## PRINT NEXT STEPS
echo ""
echo -e "\e[33mAdd augmentation to config, adjust num_examples and eval_interval_secs, and modify eval.py, eval_util.py and trainer.py"

echo ""
echo -e "\e[33mTo run training:"
echo -e "\e[97mpython research/object_detection/train.py --train_dir $FOLDERRESULT/output --pipeline_config_path $FOLDERDATA/network.config"

echo ""
echo -e "\e[33mTo run eval:"
echo -e "\e[97mpython research/object_detection/eval.py --checkpoint_dir $FOLDERRESULT/output --pipeline_config_path $FOLDERDATA/network.config  --eval_dir $FOLDERRESULT/eval"

echo ""
echo -e "\e[33mTo run tensorboard:"
echo -e "\e[97mtensorboard --logdir $FOLDERRESULT"
echo ""
echo "---------------------------"
