



# Build docker container

```bash
bash build_docker.sh
```

* This docker file clone Imagr Models repo, if you previously cache the docker container and update the repo later, want to build another docker container with the updated repo use 

```bash 
docker build --no-cache -f research/object_detection/dockerfiles/tf1/Dockerfile -t od_tf1 .
```



# Prepare tfrecord 

on nas path 

`/home/walter/nas_cv/walter_stuff/object_detection/data/3_locn_3_prods`

# Prepare Pipeline config 

* Create a dir to save all the training output, checkpoint, export model, etc. 
  * eg. my models  path: `/home/walter/nas_cv/walter_stuff/object_detection/models`
  * create  dir `det_320_320_rgb` and put the config file in here

![image-20230608105604070](./README.assets/image-20230608105604070.png)

* update the config file if needed 

# Run docker container 

in the `run_docker.sh`

Copy the tfrecord path to DATA 

Copy the model dir path to TRAINED_MODEL

```bash
export DATA="/home/walter/nas_cv/walter_stuff/object_detection/data/3_locn_3_prods"
export TRAINED_MODEL="/home/walter/nas_cv/walter_stuff/object_detection/models"
sudo docker run -it --gpus device=0 \
-v $DATA:/models/data/tfrecord \
-v $TRAINED_MODEL:/models/trained_models \
od_tf1 /bin/bash
```

mount the tfrecord to the `/models/data/tfrecord`

mount the model dir `~/git/mobileDet/saved_models/det_320_320_rgb`  to `/models/trained_models`

**Run docker container **

```bash 
bash run_docker.sh
```



# Start training 

after runing the docker 

vim the train_and_export.sh

![image-20230608110059740](./README.assets/image-20230608110059740.png)

Change the MODEL_DIR

and start training 



# tflite_convert 

After the trainning is done you should be able to get 

* tflite_graph.pb
* tflite_graph.pbtxt

![image-20230608112031373](./README.assets/image-20230608112031373.png)

create a venv 

```bash 
virtualenv venv
source venv/bin/activate 
```

Install tf2, edgetpu compiler and edgetpu runtime 

``` bash
pip install tensorflow==2.12.0

# edgetpu runtime 
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
sudo apt-get install libedgetpu1-max
sudo apt-get install python3-pycoral

# edgetpu compiler 
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
```



change the ModelDir int the `tflite_convert.sh`

`/home/walter/nas_cv/walter_stuff/object_detection/models/det_320_320_rgb`

![image-20230608115321942](./README.assets/image-20230608115321942.png)

run the script 

```bash
bash tflite_convert.sh
```

then you will get a 

`model_edgetpu.tflite` that can run on the micro controller 

