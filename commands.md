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