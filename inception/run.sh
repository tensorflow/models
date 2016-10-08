CUDA_VISIBLE_DEVICES='' bazel-bin/inception/imagenet_distributed_train \
--batch_sizet=32 \
--data_dir=/export/mfs/imagenet/ \
--job_name='ps' \
--task_id=0 \
--ps_hosts='192.168.184.39:9000' \
--worker_hosts='192.168.184.39:9001'
#--worker_hosts='192.168.184.39:9001,192.168.184.39:9002'
