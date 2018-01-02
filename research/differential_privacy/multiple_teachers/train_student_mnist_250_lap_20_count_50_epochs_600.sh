# Be sure to clone https://github.com/openai/improved-gan
# and add improved-gan/mnist_svhn_cifar10 to your PATH variable

# Download labels used to train the student
wget https://github.com/npapernot/multiple-teachers-for-privacy/blob/master/mnist_250_student_labels_lap_20.npy

# Train the student using improved-gan 
THEANO_FLAGS='floatX=float32,device=gpu,lib.cnmem=1' train_mnist_fm_custom_labels.py --labels mnist_250_student_labels_lap_20.npy --count 50 --epochs 600

