declare -A ckpt_link_map
declare -A ckpt_name_map
declare -A image_size_map
declare -A scopes_map
declare -A input_tensors_map
declare -A output_tensors_map

ckpt_link_map["mobilenet_v1"]="http://download.tensorflow.org/models/mobilenet_v1_2018_08_02/mobilenet_v1_1.0_224_quant.tgz"
ckpt_link_map["mobilenet_v2"]="http://download.tensorflow.org/models/tflite_11_05_08/mobilenet_v2_1.0_224_quant.tgz"
ckpt_link_map["inception_v1"]="http://download.tensorflow.org/models/inception_v1_224_quant_20181026.tgz"
ckpt_link_map["inception_v2"]="http://download.tensorflow.org/models/inception_v2_224_quant_20181026.tgz"
ckpt_link_map["inception_v3"]="http://download.tensorflow.org/models/tflite_11_05_08/inception_v3_quant.tgz"
ckpt_link_map["inception_v4"]="http://download.tensorflow.org/models/inception_v4_299_quant_20181026.tgz"

ckpt_name_map["mobilenet_v1"]="mobilenet_v1_1.0_224_quant"
ckpt_name_map["mobilenet_v2"]="mobilenet_v2_1.0_224_quant"
ckpt_name_map["inception_v1"]="inception_v1_224_quant"
ckpt_name_map["inception_v2"]="inception_v2_224_quant"
ckpt_name_map["inception_v3"]="inception_v3_quant"
ckpt_name_map["inception_v4"]="inception_v4_299_quant"

image_size_map["mobilenet_v1"]=224
image_size_map["mobilenet_v2"]=224
image_size_map["inception_v1"]=224
image_size_map["inception_v2"]=224
image_size_map["inception_v3"]=299
image_size_map["inception_v4"]=299

scopes_map["mobilenet_v1"]="MobilenetV1/Logits"
scopes_map["mobilenet_v2"]="MobilenetV2/Logits"
scopes_map["inception_v1"]="InceptionV1/Logits"
scopes_map["inception_v2"]="InceptionV2/Logits"
scopes_map["inception_v3"]="InceptionV3/Logits,InceptionV3/AuxLogits"
scopes_map["inception_v4"]="InceptionV4/Logits,InceptionV4/AuxLogits"

input_tensors_map["mobilenet_v1"]="input"
input_tensors_map["mobilenet_v2"]="input"
input_tensors_map["inception_v1"]="input"
input_tensors_map["inception_v2"]="input"
input_tensors_map["inception_v3"]="input"
input_tensors_map["inception_v4"]="input"

output_tensors_map["mobilenet_v1"]="MobilenetV1/Predictions/Reshape_1"
output_tensors_map["mobilenet_v2"]="MobilenetV2/Predictions/Softmax"
output_tensors_map["inception_v1"]="InceptionV1/Logits/Predictions/Softmax"
output_tensors_map["inception_v2"]="InceptionV2/Predictions/Reshape_1"
output_tensors_map["inception_v3"]="InceptionV3/Predictions/Reshape_1"
output_tensors_map["inception_v4"]="InceptionV4/Logits/Predictions"

# Change this parameter to suit your custom dataset's name
DATASET_NAME="custom"
# Assuming scripts are in tensorflow/models/research/slim/scripts folder.
SLIM_DIR="$PWD"
LEARN_DIR="../${DATASET_NAME}/learn"
CKPT_DIR="${LEARN_DIR}/ckpt"
DATASET_DIR="${LEARN_DIR}/${DATASET_NAME}"
TRAIN_DIR="${LEARN_DIR}/train"
OUTPUT_DIR="${LEARN_DIR}/models"

