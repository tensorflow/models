# From tensorflow/models
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/gdxray_rnn.pbtxt \
    --trained_checkpoint_prefix train/model.ckpt-55570 \
    --inference_graph_path output_inference_graph.pb\
    --output_directory .


# gcloud compute --project "smart-city-model" scp models/gdxray_rnn.pbtxt smart-city-collision:~/google_object/models/gdxray_rnn.pbtxt --zone "us-west1-b"

gcloud compute --project "smart-city-model" scp smart-city-collision:~/google_object/frozen_inference_graph.pb . --zone "us-west1-b"

#gcloud compute --project "smart-city-model" scp models/gdxray_pipeline.pbtxt smart-city-collision:~/google_object/odels/gdxray_pipeline.pbtxt --zone "us-west1-b" 

# TRAINING !!!
python3 object_detection/train.py \
	--logtostderr \
	--pipeline_config_path=models/rcnn_resnet.conf \
	--train_dir=train




# CLOUD TRAINING

tensorboard --logdir=gs://casting-defects/train/rfcn_resnet101/ --port 8080

gsutil acl ch -u $SVCACCT:WRITE gs://casting-defects/
gsutil defacl ch -u $SVCACCT:O gs://casting-defects/






# RFCN TRAINING AND EVALUATION
gcloud ml-engine jobs submit training casting_defects_rfcn_resnet101_`date +%s` \
    --job-dir=gs://casting-defects/train/rfcn_resnet101/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --train_dir=gs://casting-defects/train/rfcn_resnet101/ \
    --pipeline_config_path=gs://casting-defects/models/rfcn_resnet101.conf


 gcloud ml-engine jobs submit training casting_defects_rcnn_resnet101_eval_`date +%s` \
    --job-dir=gs://casting-defects/train/rfcn_resnet101/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --eval_dir=gs://casting-defects/eval/rfcn_resnet101/ \
    --checkpoint_dir=gs://casting-defects/train/rfcn_resnet101/
    --pipeline_config_path=gs://casting-defects/models/rfcn_resnet101.conf




 # ---->>    <<------ Faster RCNN ResNet101 Training and Evaluation
gcloud ml-engine jobs submit training casting_defects_faster_rcnn_resnet101_`date +%s` \
    --job-dir=gs://casting-defects/train/faster_rcnn_resnet101/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --train_dir=gs://casting-defects/train/faster_rcnn_resnet101/ \
    --pipeline_config_path=gs://casting-defects/models/faster_rcnn_resnet101.conf


 gcloud ml-engine jobs submit training casting_defects_rcnn_resnet101_eval_`date +%s` \
    --job-dir=gs://casting-defects/train/faster_rcnn_resnet101/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --eval_dir=gs://casting-defects/eval/faster_rcnn_resnet101/ \
    --checkpoint_dir=gs://casting-defects/train/faster_rcnn_resnet101/
    --pipeline_config_path=gs://casting-defects/models/faster_rcnn_resnet101.conf




# Faster RCNN Inception Training and Evaluation
gcloud ml-engine jobs submit training casting_defects_faster_rcnn_inception_`date +%s` \
    --job-dir=gs://casting-defects/train/faster_rcnn_inception/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --train_dir=gs://casting-defects/train/faster_rcnn_inception/ \
    --pipeline_config_path=gs://casting-defects/models/faster_rcnn_inception.conf


 gcloud ml-engine jobs submit training casting_defects_rcnn_inception_eval_`date +%s` \
    --job-dir=gs://casting-defects/train/faster_rcnn_inception/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --eval_dir=gs://casting-defects/eval/faster_rcnn_inception/ \
    --checkpoint_dir=gs://casting-defects/train/faster_rcnn_inception/
    --pipeline_config_path=gs://casting-defects/models/faster_rcnn_inception.conf




# SSD Inception Training and Evaluation
gcloud ml-engine jobs submit training casting_defects_ssd_inception_`date +%s` \
    --job-dir=gs://casting-defects/train/ssd_inception/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --train_dir=gs://casting-defects/train/ssd_inception/ \
    --pipeline_config_path=gs://casting-defects/models/ssd_inception.conf


 gcloud ml-engine jobs submit training casting_defects_ssd_inception_eval_`date +%s` \
    --job-dir=gs://casting-defects/train/ssd_inception/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --eval_dir=gs://casting-defects/eval/ssd_inception/ \
    --checkpoint_dir=gs://casting-defects/train/ssd_inception/
    --pipeline_config_path=gs://casting-defects/models/ssd_inception.conf





# SSD Mobilenet Training and Evaluation
gcloud ml-engine jobs submit training casting_defects_ssd_mobilenet_`date +%s` \
    --job-dir=gs://casting-defects/train/ssd_mobilenet/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --train_dir=gs://casting-defects/train/ssd_mobilenet/ \
    --pipeline_config_path=gs://casting-defects/models/ssd_mobilenet.conf


 gcloud ml-engine jobs submit training casting_defects_ssd_mobilenet_eval_`date +%s` \
    --job-dir=gs://casting-defects/train/ssd_mobilenet/ \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --eval_dir=gs://casting-defects/eval/ssd_mobilenet/ \
    --checkpoint_dir=gs://casting-defects/train/ssd_mobilenet/
    --pipeline_config_path=gs://casting-defects/models/ssd_mobilenet.conf
