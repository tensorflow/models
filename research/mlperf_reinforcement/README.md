# 1. Problem 
This task benchmarks on policy reinforcement learning for the 9x9 version of the boardgame go. The model plays games against itself and uses these games to improve play.

# 2. Directions
### Steps to configure machine
To setup the environment on Ubuntu 16.04 (16 CPUs, one P100, 100 GB disk), you can use these commands. This may vary on a different operating system or graphics card.

    # Install docker
    sudo apt-get install -y apt-transport-https ca-certificates curl software-properties-common

    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo apt-key fingerprint 0EBFCD88
    sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
       $(lsb_release -cs) \
       stable"
    sudo apt update
    # sudo apt install docker-ce -y
    sudo apt install docker-ce=18.03.0~ce-0~ubuntu -y --allow-downgrades

    # Install nvidia-docker2
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey |   sudo apt-key add -
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu16.04/nvidia-docker.list |   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
    sudo apt-get update
    sudo apt install nvidia-docker2 -y


    sudo tee /etc/docker/daemon.json <<EOF
    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            }
        }
    }
    EOF
    sudo pkill -SIGHUP dockerd

    sudo apt install -y bridge-utils
    sudo service docker stop
    sleep 1;
    sudo iptables -t nat -F
    sleep 1;
    sudo ifconfig docker0 down
    sleep 1;
    sudo brctl delbr docker0
    sleep 1;
    sudo service docker start

    ssh-keyscan github.com >> ~/.ssh/known_hosts
    git clone git@github.com:mlperf/reference.git

### Steps to download and verify data
Unlike other benchmarks, there is no data to download. All training data comes from games played during benchmarking.

### Steps to run and time

To run, this assumes you checked out the repo into $HOME, adjust paths as necessary.

    cd ~/reference/reinforcement/tensorflow/
    IMAGE=`sudo docker build . | tail -n 1 | awk '{print $3}'`
    SEED=1
    NOW=`date "+%F-%T"`
    sudo docker run --runtime=nvidia -t -i $IMAGE "./run_and_time.sh" $SEED | tee benchmark-$NOW.log
    
To change the quality target, modify `params/final.json` and set the field `TERMINATION_ACCURACY` to be `0.10` for about a 10 hour runtime, or `0.03` for about a 3 hour runtime. Note that you will have to rebuild the docker after modifying `params/final.josn`.

# 3. Model
### Publication/Attribution

This benchmark is based on a fork of the minigo project (https://github.com/tensorflow/minigo); which is inspired by the work done by Deepmind with ["Mastering the Game of Go with Deep Neural Networks and
Tree Search"](https://www.nature.com/articles/nature16961), ["Mastering the Game of Go without Human
Knowledge"](https://www.nature.com/articles/nature24270), and ["Mastering Chess and Shogi by
Self-Play with a General Reinforcement Learning
Algorithm"](https://arxiv.org/abs/1712.01815). Note that minigo is an
independent effort from AlphaGo, and that this fork is minigo is independent from minigo itself. 

### Reinforcement Setup

This benchmark includes both the environment and training for 9x9 go. There are four primary phases in this benchmark, these phases are repeated in order:

 - Selfplay: the *current best* model plays games against itself to produce board positions for training.
 - Training: train the neural networks selfplay data from several recent models. 
 - Target Evaluation: the termination criteria for the benchmark is checked using the provided record of professional games. 
 - Model Evaluation: the *current best* and the most recently trained model play a series of games. In order to become the new *current best*, the most recently trained model must win 55% of the games.

### Structure

This task has a non-trivial network structure, including a search tree. A good overview of the structure can be found here: https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0. 

### Weight and bias initialization and Loss Function
Network weights are initialized randomly. Initialization and loss are described here;
["Mastering the Game of Go with Deep Neural Networks and Tree Search"](https://www.nature.com/articles/nature16961)

### Optimizer
We use a MomentumOptimizer to train the primary network. 

# 4. Quality

Due to the difficulty of training a highly proficient go model, our quality metric and termination criteria is based on predicting moves from human reference games. Currently published results indicate that it takes weeks of time and/or cluster sized resources to achieve a high level of play. Given more limited time and resources, it is possible to predict a significant number of moves from professional or near-professional games. 

### Quality metric

Provided in with this benchmark are records of human games and the quality metric is the percent of the time the model chooses the same move the human chose in each position. Each position is attempted twice by the model (keep in mind the model's choice is non-deterministic). The metric is calculated as the number of correct predictions divided by the number of predictions attempted. 

The particular games we use are from Iyama Yuta 6 Title Celebration, between contestants Murakawa Daisuke, Sakai Hideyuki, Yamada Kimio, Hyakuta Naoki, Yuki Satoshi, and Iyama Yuta.



### Quality target
The quality target is predicting 40% of the moves.

### Quality Progression
Informally, we have observed that quality should improve roughly linearly with time. We observed roughly 0.5% improvement in quality per hour of runtime. An example of approximately how we've seen quality progress over time:

    Approx. Hours to Quality (16 CPU & 1 P100)
    2h           3%
    12h          14%
    24h          19%
    36h          24%
    60h          34%

Note that quality does not necessarily monotonically increase. 

### Evaluation frequency
Evaluation should be preformed for every model which is trained (regardless if it wins the "model evaluation" round). 
    

### Evaluation thoroughness
All positions should be considered in each evaluation phase.
