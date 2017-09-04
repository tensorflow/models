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

gsutil acl ch -u $SVCACCT:WRITE gs://casting-defects/
gsutil defacl ch -u $SVCACCT:O gs://casting-defects/

# From casting-defects
gcloud ml-engine jobs submit training casting_defects_rcnn_resnet101_`date +%s` \
    --job-dir=gs://casting-defects/train \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.train \
    --region us-east1 \
    --config google_cloud.yaml \
    -- \
    --train_dir=gs://casting-defects/train \
    --pipeline_config_path=gs://casting-defects/models/rcnn_resnet-101.conf


 gcloud ml-engine jobs submit training asting_defects_rcnn_resnet101_eval_`date +%s` \
    --job-dir=gs://casting-defects/train \
    --packages dist/object_detection-0.1.tar.gz,object_detection/slim/dist/slim-0.1.tar.gz \
    --module-name object_detection.eval \
    --region us-east1 \
    --scale-tier BASIC_GPU \
    -- \
    --job-dir=gs://casting-defects/train \
    --eval_dir=gs://casting-defects/eval \
    --pipeline_config_path=gs://casting-defects/models/rcnn_resnet-101.conf