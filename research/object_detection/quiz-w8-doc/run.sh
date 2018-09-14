#!/bin/bash
# 查找脚本所在路径，并进入
#DIR="$( cd "$( dirname "$0"  )" && pwd  )"
DIR=$PWD
cd $DIR
echo current dir is $PWD

# 设置目录，避免module找不到的问题
export PYTHONPATH=$PYTHONPATH:$DIR:$DIR/slim:$DIR/object_detection

# 定义各目录
output_dir=/output  # 训练目录
dataset_dir=/data/ai100/quiz-w8 # 数据集目录，这里是写死的，记得修改

train_dir=$output_dir/train
checkpoint_dir=$train_dir
eval_dir=$output_dir/eval

# config文件
config=ssd_mobilenet_v1_pets.config
pipeline_config_path=$output_dir/$config

# 先清空输出目录，本地运行会有效果，tinymind上运行这一行没有任何效果
# tinymind已经支持引用上一次的运行结果，这一行需要删掉，不然会出现上一次的运行结果被清空的状况。
# rm -rvf $output_dir/*

# 因为dataset里面的东西是不允许修改的，所以这里要把config文件复制一份到输出目录
cp $dataset_dir/$config $pipeline_config_path

for i in {0..4}  # for循环中的代码执行5此，这里的左右边界都包含，也就是一共训练500个step，每100step验证一次
do
    echo "############" $i "runnning #################"
    last=$[$i*100]
    current=$[($i+1)*100]
    sed -i "s/^  num_steps: $last$/  num_steps: $current/g" $pipeline_config_path  # 通过num_steps控制一次训练最多100step

    echo "############" $i "training #################"
    python ./object_detection/train.py --train_dir=$train_dir --pipeline_config_path=$pipeline_config_path

    echo "############" $i "evaluating, this takes a long while #################"
    python ./object_detection/eval.py --checkpoint_dir=$checkpoint_dir --eval_dir=$eval_dir --pipeline_config_path=$pipeline_config_path
done

# 导出模型
python ./object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path $pipeline_config_path --trained_checkpoint_prefix $train_dir/model.ckpt-$current  --output_directory $output_dir/exported_graphs

# 在test.jpg上验证导出的模型
python ./inference.py --output_dir=$output_dir --dataset_dir=$dataset_dir
