export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
python train.py --logtostderr --train_dir=data/ --pipeline=data/dog-train.config

