# Image Compression with Neural Networks

This is a [TensorFlow](http://www.tensorflow.org/) model for compressing and
decompressing images using an already trained  Residual GRU model as descibed
in [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148). Please consult the paper for more details
on the architecture and compression results.

This code will allow you to perform the lossy compression on an model
already trained on compression. This code doesn't not currently contain the
Entropy Coding portions of our paper.


## Prerequisites
The only software requirements for running the encoder and decoder is having
Tensorflow installed. You will also need to [download](http://download.tensorflow.org/models/compression_residual_gru-2016-08-23.tar.gz)
and extract the model residual_gru.pb.

If you want to generate the perceptual similarity under MS-SSIM, you will also
need to [Install SciPy](https://www.scipy.org/install.html).

## Encoding
The Residual GRU network is fully convolutional, but requires the images
height and width in pixels by a multiple of 32. There is an image in this folder
called example.png that is 768x1024 if one is needed for testing. We also
rely on TensorFlow's built in decoding ops, which support only PNG and JPEG at
time of release.

To encode an image, simply run the following command:

`python encoder.py --input_image=/your/image/here.png
--output_codes=output_codes.npz --iteration=15
--model=/path/to/model/residual_gru.pb
`

The iteration parameter specifies the lossy-quality to target for compression.
The quality can be [0-15], where 0 corresponds to a target of 1/8 (bits per
pixel) bpp and every increment results in an additional 1/8 bpp.

| Iteration | BPP | Compression Ratio |
|---: |---: |---: |
|0 | 0.125 | 192:1|
|1 | 0.250 | 96:1|
|2 | 0.375 | 64:1|
|3 | 0.500 | 48:1|
|4 | 0.625 | 38.4:1|
|5 | 0.750 | 32:1|
|6 | 0.875 | 27.4:1|
|7 | 1.000 | 24:1|
|8 | 1.125 | 21.3:1|
|9 | 1.250 | 19.2:1|
|10 | 1.375 | 17.4:1|
|11 | 1.500 | 16:1|
|12 | 1.625 | 14.7:1|
|13 | 1.750 | 13.7:1|
|14 | 1.875 | 12.8:1|
|15 | 2.000 | 12:1|

The output_codes file contains the numpy shape and a flattened, bit-packed
array of the codes. These can be inspected in python by using numpy.load().


## Decoding
After generating codes for an image, the lossy reconstructions for that image
can be done as follows:

`python decoder.py --input_codes=codes.npz --output_directory=/tmp/decoded/
--model=residual_gru.pb`

The output_directory will contain images decoded at each quality level.


## Comparing Similarity
One of our primary metrics for comparing how similar two images are
is MS-SSIM.

To generate these metrics on your images you can run:
`python msssim.py --original_image=/path/to/your/image.png
--compared_image=/tmp/decoded/image_15.png`


## Results
CSV results containing the post-entropy bitrates and MS-SSIM over Kodak can 
are available for reference. Each row of the CSV represents each of the Kodak
images in their dataset number (1-24). Each column of the CSV represents each
iteration of the model (1-16).

[Post Entropy Bitrates](https://storage.googleapis.com/compression-ml/residual_gru_results/bitrate.csv)

[MS-SSIM](https://storage.googleapis.com/compression-ml/residual_gru_results/msssim.csv)


## FAQ

#### How do I train my own compression network?
We currently don't provide the code to build and train a compression
graph from scratch.

#### I get an InvalidArgumentError: Incompatible shapes.
This is usually due to the fact that our network only supports images that are
both height and width divisible by 32 pixel. Try padding your images to 32
pixel boundaries.


## Contact Info
Model repository maintained by Nick Johnston ([nmjohn](https://github.com/nmjohn)).
