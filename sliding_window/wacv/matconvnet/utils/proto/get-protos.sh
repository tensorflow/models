# FCN
wget -nc "https://raw.githubusercontent.com/longjon/caffe/6e3916766c6b63bff07e2cfadf210ee5e46af807/src/caffe/proto/caffe.proto" --output-document=./caffe_6e3916.proto
protoc ./caffe_6e3916.proto --python_out=./

# b590f1d (ResNet)
wget -nc "https://raw.githubusercontent.com/BVLC/caffe/b590f1d27eb5cbd9bc7b9157d447706407c68682/src/caffe/proto/caffe.proto" --output-document=./caffe_b590f1d.proto
protoc ./caffe_b590f1d.proto --python_out=./
