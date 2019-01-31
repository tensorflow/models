Automating the Evaluation of Crystallization Experiments
========================================================

This is a pretrained model described in the paper:

[Classification of crystallization outcomes using deep convolutional neural networks](https://arxiv.org/abs/1803.10342).

This model takes images of crystallization experiments as an input:

<img src="https://storage.googleapis.com/marco-168219-model/002s_C6_ImagerDefaults_9.jpg" alt="crystal sample" width="320" height="240" />

It classifies it as belonging to one of four categories: crystals, precipitate, clear, or 'others'.

The model is a variant of [Inception-v3](https://arxiv.org/abs/1512.00567) trained on data from the [MARCO](http://marco.ccr.buffalo.edu) repository.

Model
-----

The model can be downloaded from:

https://storage.googleapis.com/marco-168219-model/savedmodel.zip

Example
-------

1. Install TensorFlow and the [Google Cloud SDK](https://cloud.google.com/sdk/gcloud/).

2. Download and unzip the model:

 ```bash
 unzip savedmodel.zip
 ```

3. A sample image can be downloaded from:

 https://storage.googleapis.com/marco-168219-model/002s_C6_ImagerDefaults_9.jpg

 Convert your image into a JSON request using:

 ```bash
 python jpeg2json.py 002s_C6_ImagerDefaults_9.jpg > request.json
 ```

4. To issue a prediction, run:

 ```bash
 gcloud ml-engine local predict --model-dir=savedmodel --json-instances=request.json
 ```

The request should return normalized scores for each class:

<pre>
CLASSES                                            SCORES
[u'Crystals', u'Other', u'Precipitate', u'Clear']  [0.926338255405426, 0.026199858635663986, 0.026074528694152832, 0.021387407556176186]
</pre>

CloudML Endpoint
----------------

The model can also be accessed on [Google CloudML](https://cloud.google.com/ml-engine/) by issuing:

```bash
gcloud ml-engine predict --model marco_168219_model --json-instances request.json
```

Ask the author for access privileges to the CloudML instance.

Note
----

`002s_C6_ImagerDefaults_9.jpg` is a sample from the
[MARCO](http://marco.ccr.buffalo.edu) repository, contributed to the dataset under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license.

Author
------

[Vincent Vanhoucke](mailto:vanhoucke@google.com) (github: vincentvanhoucke)
                                                                                
