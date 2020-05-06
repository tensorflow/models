![Logo](https://storage.googleapis.com/model_garden_artifacts/TF_Model_Garden.png)

# TensorFlow Research Models

This directory contains code implementations and pre-trained models of published research papers.

The research models are maintained by their respective authors.

## Table of Contents
- [Modeling Libraries and Models](#modeling-libraries-and-models)
- [Models and Implementations](#models-and-implementations)
  * [Computer Vision](#computer-vision)
  * [Natural Language Processing](#natural-language-processing)
  * [Audio and Speech](#audio-and-speech)
  * [Reinforcement Learning](#reinforcement-learning)
  * [Others](#others)
- [Archived Models and Implementations](#warning-archived-models-and-implementations) (:no_entry_sign: No longer maintained)

## Modeling Libraries and Models

| Directory | Name | Description | Maintainer(s) |
|-----------|------|-------------|---------------|
| [object_detection](object_detection) | TensorFlow Object Detection API | A framework that makes it easy to construct, train and deploy object detection models<br /><br />A collection of object detection models pre-trained on the COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset, and the iNaturalist Species Detection Dataset| @jch1, @tombstone, @pkulzc |
| [slim](slim) | TensorFlow-Slim Image Classification Model Library | A lightweight high-level API of TensorFlow for defining, training and evaluating image classification models <br />• Inception V1/V2/V3/V4<br />• Inception-ResNet-v2<br />• ResNet V1/V2<br />• VGG 16/19<br />• MobileNet V1/V2/V3<br />• NASNet-A_Mobile/Large<br />• PNASNet-5_Large/Mobile | @sguada, @marksandler2 |

## Models and Implementations

### Computer Vision

| Directory | Referenece (Paper) | Maintainer(s) |
|-----------|--------------------|---------------|
| [attention_ocr](attention_ocr) | [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/abs/1704.03549) | xavigibert |
| [autoaugment](autoaugment) | [1] [AutoAugment](https://arxiv.org/abs/1805.09501)<br />[2] [Wide Residual Networks](https://arxiv.org/abs/1605.07146)<br />[3] [Shake-Shake regularization](https://arxiv.org/abs/1705.07485)<br />[4] [ShakeDrop Regularization for Deep Residual Learning](https://arxiv.org/abs/1802.02375) | barretzoph |
| [deeplab](deeplab) | [1] [DeepLabv1](https://arxiv.org/abs/1412.7062)<br />[2] [DeepLabv2](https://arxiv.org/abs/1606.00915)<br />[3] [DeepLabv3](https://arxiv.org/abs/1802.02611)<br />[4] [DeepLabv3+](https://arxiv.org/abs/1706.05587) | aquariusjay, yknzhu |
| [delf](delf)  | [1] DELF (DEep Local Features): [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321)<br />[2] [Detect-to-Retrieve](https://arxiv.org/abs/1812.01584) | andrefaraujo |
| [lstm_object_detection](lstm_object_detection) | [Mobile Video Object Detection with Temporally-Aware Feature Maps](https://arxiv.org/abs/1711.06368) | yinxiaoli, yongzhe2160, lzyuan |
| [marco](marco) | [Classification of crystallization outcomes using deep convolutional neural networks](https://arxiv.org/abs/1803.10342) | vincentvanhoucke |
| [vid2depth](vid2depth) | [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522) | rezama |

### Natural Language Processing

| Directory | Referenece (Paper) | Maintainer(s) |
|-----------|--------------------|---------------|
| [adversarial_text](adversarial_text) | [1] [Adversarial Training Methods for Semi-Supervised Text](https://arxiv.org/abs/1605.07725) Classification<br />[2] [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) | rsepassi, a-dai |
| [cvt_text](cvt_text) | [Semi-supervised sequence learning with cross-view training](https://arxiv.org/abs/1809.08370) | clarkkev, lmthang |

### Audio and Speech

| Directory | Referenece (Paper) | Maintainer(s) |
|-----------|--------------------|---------------|
| [audioset](audioset) | [1] [AudioSet: A Large Scale Dataset of Audio Events](https://research.google/pubs/pub45857/)<br />[2] [CNN Architectures for Large-Scale Audio Classification](https://research.google/pubs/pub45611/) | plakal, dpwe |

### Reinforcement Learning

| Directory | Referenece (Paper) | Maintainer(s) |
|-----------|--------------------|---------------|
| [efficient-hrl](efficient-hrl) | [1] [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)<br />[2] [Near-Optimal Representation Learning for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1810.01257) | ofirnachum |
| [pcl_rl](pcl_rl) | [1] [Improving Policy Gradient by Exploring Under-appreciated Rewards](https://arxiv.org/abs/1611.09321)<br />[2] [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)<br />[3] [Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891) | ofirnachum |

### Others

| Directory | Referenece (Paper) | Maintainer(s) |
|-----------|--------------------|---------------|
| [lfads](lfads) | [LFADS - Latent Factor Analysis via Dynamical Systems](https://doi.org/10.1101/152884) | jazcollins, sussillo |
| [rebar](rebar) | [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370) | gjtucker |

---

## :warning: Archived Models and Implementations

The following research models are no longer maintained.

**Note**: We will remove archived models from the master branch in June, 2020. 
After removal, you will still be able to access archived models in the archive branch.

| Directory | Referenece (Paper) | Maintainer(s) |
|-----------|--------------------|---------------|
| [adv_imagenet_models](adv_imagenet_models) | [1] [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)<br />[2] [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) | alexeykurakin |
| [adversarial_crypto](adversarial_crypto) | [Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918) | dave-andersen |
| [adversarial_logit_pairing](adversarial_logit_pairing) | [Adversarial Logit Pairing](https://arxiv.org/abs/1803.06373) | alexeykurakin |
| [autoencoder](autoencoder) | Various autoencoders | snurkabill |
| [brain_coder](brain_coder) | [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526) | danabo, mnorouzi |
| [cognitive_mapping_and_planning](cognitive_mapping_and_planning) | [Cognitive Mapping and Planning for Visual Navigation](https://arxiv.org/abs/1702.03920) | s-gupta |
| [compression](compression) | [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148) | nmjohn |
| [deep_contextual_bandits](deep_contextual_bandits) | [Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127) | rikel |
| [deep_speech](deep_speech) | [Deep Speech 2](https://arxiv.org/abs/1512.02595) | yhliang2018 |
| [domain_adaptation](domain_adaptation) | [1] [Domain Separation Networks](https://arxiv.org/abs/1608.06019) <br />[2] [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1612.05424) | bousmalis, dmrd |
| [feelvos](feelvos)| [FEELVOS](https://arxiv.org/abs/1902.09513) | pvoigtlaender, yuningchai, aquariusjay |
| [fivo](fivo)| [Filtering variational objectives for training generative sequence models](https://arxiv.org/abs/1705.09279) | dieterichlawson |
| [global_objectives](global_objectives) | [Scalable Learning of Non-Decomposable Objectives](https://arxiv.org/abs/1608.04802) | mackeya-google |
| [im2txt](im2txt) | [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge](https://arxiv.org/abs/1609.06647) | cshallue |
| [inception](inception) | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | shlens, vincentvanhoucke |
| [keypointnet](keypointnet) | [KeypointNet](https://arxiv.org/abs/1807.03146) | mnorouzi |
| [learned_optimizer](learned_optimizer) | [Learned Optimizers that Scale and Generalize](https://arxiv.org/abs/1703.04813) | olganw, nirum |
| [learning_to_remember_rare_events](learning_to_remember_rare_events) | [Learning to Remember Rare Events](https://arxiv.org/abs/1703.03129) | lukaszkaiser, ofirnachum |
| [learning_unsupervised_learning](learning_unsupervised_learning) | [Meta-Learning Update Rules for Unsupervised Representation Learning](https://arxiv.org/abs/1804.00222) | lukemetz, nirum |
| [lexnet_nc](lexnet_nc) | [Olive Oil is Made of Olives, Baby Oil is Made for Babies: Interpreting Noun Compounds using Paraphrases in a Neural Model](https://arxiv.org/abs/1803.08073) | vered1986, waterson |
| [lm_1b](lm_1b) | [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) | oriolvinyals, panyx0718 |
| [lm_commonsense](lm_commonsense) | [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847) | thtrieu |
| [maskgan](maskgan)| [MaskGAN: Better Text Generation via Filling in the______](https://arxiv.org/abs/1801.07736) | liamb315, a-dai |
| [namignizer](namignizer)| Namignizer | knathanieltucker |
| [neural_gpu](neural_gpu)| [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) | lukaszkaiser |
| [neural_programmer](neural_programmer) | [Learning a Natural Language Interface with Neural Programmer](https://arxiv.org/abs/1611.08945) | arvind2505 |
| [next_frame_prediction](next_frame_prediction) | [Visual Dynamics](https://arxiv.org/abs/1607.02586) | panyx0718 |
| [ptn](ptn) | [Perspective Transformer Nets](https://arxiv.org/abs/1612.00814) | xcyan, arkanath, hellojas, honglaklee |
| [qa_kg](qa_kg) | [Learning to Reason](https://arxiv.org/abs/1704.05526) | yuyuz |
| [real_nvp](real_nvp) | [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) | laurent-dinh |
| [sentiment_analysis](sentiment_analysis)| [Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://arxiv.org/abs/1412.1058) | sculd |
| [seq2species](seq2species) | [Seq2Species: A deep learning approach to pattern recognition for short DNA sequences](https://doi.org/10.1101/353474) | apbusia, depristo |
| [skip_thoughts](skip_thoughts) | [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726) | cshallue |
| [steve](steve) | [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675) | buckman-google |
| [street](street) | [End-to-End Interpretation of the French Street Name Signs Dataset](https://arxiv.org/abs/1702.03970) | theraysmith |
| [struct2depth](struct2depth)| [Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/abs/1811.06152) | aneliaangelova |
| [swivel](swivel) | [Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/abs/1602.02215) | waterson |
| [tcn](tcn) | [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888) | coreylynch, sermanet |
| [textsum](textsum)| [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685) | panyx0718, peterjliu |
| [transformer](transformer) | [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) | daviddao|
| [video_prediction](video_prediction) | [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157) | cbfinn |

---

## Contributions

If you want to contribute, please review the [contribution guidelines](../../../wiki/How-to-contribute).
