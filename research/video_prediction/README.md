# Video Prediction with Neural Advection

*A TensorFlow implementation of the models described in [Unsupervised Learning for Physical Interaction through Video Prediction (Finn et al., 2016)](https://arxiv.org/abs/1605.07157).*

This video prediction model, which is optionally conditioned on actions,
predictions future video by internally predicting how to transform the last
image (which may have been predicted) into the next image. As a result, it can
reuse apperance information from previous frames and can better generalize to
objects not seen in the training set. Some example predictions on novel objects
are shown below:

![Animation](https://storage.googleapis.com/push_gens/novelgengifs9/16_70.gif)
![Animation](https://storage.googleapis.com/push_gens/novelgengifs9/2_96.gif)
![Animation](https://storage.googleapis.com/push_gens/novelgengifs9/1_38.gif)
![Animation](https://storage.googleapis.com/push_gens/novelgengifs9/11_10.gif)
![Animation](https://storage.googleapis.com/push_gens/novelgengifs9/3_34.gif)

When the model is conditioned on actions, it changes it's predictions based on
the passed in action. Here we show the models predictions in response to varying
the magnitude of the passed in actions, from small to large:

![Animation](https://storage.googleapis.com/push_gens/webgifs/0xact_0.gif)
![Animation](https://storage.googleapis.com/push_gens/05xact_0.gif)
![Animation](https://storage.googleapis.com/push_gens/webgifs/1xact_0.gif)
![Animation](https://storage.googleapis.com/push_gens/webgifs/15xact_0.gif)

![Animation](https://storage.googleapis.com/push_gens/webgifs/0xact_17.gif)
![Animation](https://storage.googleapis.com/push_gens/webgifs/05xact_17.gif)
![Animation](https://storage.googleapis.com/push_gens/webgifs/1xact_17.gif)
![Animation](https://storage.googleapis.com/push_gens/webgifs/15xact_17.gif)


Because the model is trained with an l2 objective, it represents uncertainty as
blur.

## Stochastic Variational Video Prediction (SV2P)
* A TensorFlow implementation of [Stochastic Variational Video Prediction (Babaeizadeh et al., 2017)](https://arxiv.org/abs/1710.11252).

This model is essentially a stochastic version of the previous model, which improves
the quality of predicted video in stochastic and unpredictable environments by removing
the blur mentioned above. The newly added section of the model approximates the posterior
that will be used by the modified original model to predict the motion.

To view samples of the stochastic version, please check [this website](https://sites.google.com/site/stochasticvideoprediction/).


## Requirements
* Tensorflow (see tensorflow.org for installation instructions)
* spatial_tranformer model in tensorflow/models, for the spatial tranformer
  predictor (STP).

## Data
The data used to train this model is located
[here](https://sites.google.com/site/brainrobotdata/home/push-dataset).

To download the robot data, run the following.
```shell
./download_data.sh
```

## Training the model

To train the model, run the prediction_train.py file.
```shell
python prediction_train.py
```

There are several flags which can control the model that is trained, which are
exeplified below:
```shell
python prediction_train.py \
  --data_dir=push/push_train \ # path to the training set.
  --model=CDNA \ # the model type to use - DNA, CDNA, or STP
  --output_dir=./checkpoints \ # where to save model checkpoints
  --event_log_dir=./summaries \ # where to save training statistics
  --num_iterations_1st_stage=100000 \ # number of training iterations
  --pretrained_model=model \ # path to model to initialize from, random if emtpy
  --sequence_length=10 \ # the number of total frames in a sequence
  --context_frames=2 \ # the number of ground truth frames to pass in at start
  --use_state=1 \ # whether or not to condition on actions and the initial state
  --num_masks=10 \ # the number of transformations and corresponding masks
  --schedsamp_k=900.0 \ # the constant used for scheduled sampling or -1
  --train_val_split=0.95 \ # the percentage of training data for validation
  --batch_size=32 \ # the training batch size
  --learning_rate=0.001 \ # the initial learning rate for the Adam optimizer
```

If the dynamic neural advection (DNA) model is being used, the `--num_masks`
option should be set to one.

The `--context_frames` option defines both the number of initial ground truth
frames to pass in, as well as when to start penalizing the model's predictions.

The data directory `--data_dir` should contain tfrecord files with the format
used in the released push dataset. See
[here](https://sites.google.com/site/brainrobotdata/home/push-dataset) for
details. If the `--use_state` option is not set, then the data only needs to
contain image sequences, not states and actions.


The stochastic version of the model exposes a handful of new arguments:
```shell
  --stochastic_model=True \ # to enable the stochastic model
  --inference_time=False \ # to use random latents at inference time
  --multi_latent=False \ # to use time-variant latent
  --latent_std_min=-5.0 \ # minimum value for standard deviation of posterior
  --latent_loss_multiplier=1e-3 \ # beta value which adjusts the KL loss
  --latent_channels=1 \ # number of latent channels
  --num_iterations_1st_stage=100000 \ # number of iterations for 1st stage of training
  --num_iterations_2nd_stage=50000 \ # number of iterations for 2nd stage of training
  --num_iterations_3rd_stage=50000 \ # number of iterations for 3rd stage of training
```

## Contact

To ask questions or report issues please open an issue on the tensorflow/models
[issues tracker](https://github.com/tensorflow/models/issues).
Please assign issues to @cbfinn and @mbz.

## Credits

This code was written by Chelsea Finn.
The stochastic addon was coded by Mohammad Babaeizadeh.
