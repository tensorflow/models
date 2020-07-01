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
| [object_detection](object_detection) | TensorFlow Object Detection API | A framework that makes it easy to construct, train and deploy object detection models<br /><br />A collection of object detection models pre-trained on the COCO dataset, the Kitti dataset, the Open Images dataset, the AVA v2.1 dataset, and the iNaturalist Species Detection Dataset| jch1, tombstone, pkulzc |
| [slim](slim) | TensorFlow-Slim Image Classification Model Library | A lightweight high-level API of TensorFlow for defining, training and evaluating image classification models <br />• Inception V1/V2/V3/V4<br />• Inception-ResNet-v2<br />• ResNet V1/V2<br />• VGG 16/19<br />• MobileNet V1/V2/V3<br />• NASNet-A_Mobile/Large<br />• PNASNet-5_Large/Mobile | sguada, marksandler2 |

## Models and Implementations

### Computer Vision

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [attention_ocr](attention_ocr) | [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/abs/1704.03549) | ICDAR 2017 | xavigibert |
| [autoaugment](autoaugment) | [1] [AutoAugment](https://arxiv.org/abs/1805.09501)<br />[2] [Wide Residual Networks](https://arxiv.org/abs/1605.07146)<br />[3] [Shake-Shake regularization](https://arxiv.org/abs/1705.07485)<br />[4] [ShakeDrop Regularization for Deep Residual Learning](https://arxiv.org/abs/1802.02375) | [1] CVPR 2019<br />[2] BMVC 2016<br /> [3] ICLR 2017<br /> [4] ICLR 2018 | barretzoph |
| [deeplab](deeplab) | [1] [DeepLabv1: Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/abs/1412.7062)<br />[2] [DeepLabv2: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/abs/1606.00915)<br />[3] [DeepLabv3: Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)<br />[4] [DeepLabv3+: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611)<br />| [1] ICLR 2015 <br />[2] TPAMI 2017 <br />[4] ECCV 2018 | aquariusjay, yknzhu |
| [delf](delf)  | [1] DELF (DEep Local Features): [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321)<br />[2] [Detect-to-Retrieve: Efficient Regional Aggregation for Image Search](https://arxiv.org/abs/1812.01584)<br />[3] DELG (DEep Local and Global features): [Unifying Deep Local and Global Features for Image Search](https://arxiv.org/abs/2001.05027)<br />[4] GLDv2: [Google Landmarks Dataset v2 -- A Large-Scale Benchmark for Instance-Level Recognition and Retrieval](https://arxiv.org/abs/2004.01804) | [1] ICCV 2017<br />[2] CVPR 2019<br />[4] CVPR 2020 | andrefaraujo |
| [lstm_object_detection](lstm_object_detection) | [Mobile Video Object Detection with Temporally-Aware Feature Maps](https://arxiv.org/abs/1711.06368) | CVPR 2018 | yinxiaoli, yongzhe2160, lzyuan |
| [marco](marco) | MARCO: [Classification of crystallization outcomes using deep convolutional neural networks](https://arxiv.org/abs/1803.10342) | | vincentvanhoucke |
| [vid2depth](vid2depth) | [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522) | CVPR 2018 | rezama |

### Natural Language Processing

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [adversarial_text](adversarial_text) | [1] [Adversarial Training Methods for Semi-Supervised Text](https://arxiv.org/abs/1605.07725) Classification<br />[2] [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) | [1] ICLR 2017<br />[2] NIPS 2015 | rsepassi, a-dai |
| [cvt_text](cvt_text) | [Semi-Supervised Sequence Modeling with Cross-View Training](https://arxiv.org/abs/1809.08370) | EMNLP 2018 | clarkkev, lmthang |

### Audio and Speech

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [audioset](audioset) | [1] [Audio Set: An ontology and human-labeled dataset for audio events](https://research.google/pubs/pub45857/)<br />[2] [CNN Architectures for Large-Scale Audio Classification](https://research.google/pubs/pub45611/) | ICASSP 2017 | plakal, dpwe |

### Reinforcement Learning

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [efficient-hrl](efficient-hrl) | [1] [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)<br />[2] [Near-Optimal Representation Learning for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1810.01257) | [1] NIPS 2018<br /> [2] ICLR 2019 | ofirnachum |
| [pcl_rl](pcl_rl) | [1] [Improving Policy Gradient by Exploring Under-appreciated Rewards](https://arxiv.org/abs/1611.09321)<br />[2] [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)<br />[3] [Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891) | [1] ICLR 2017<br />[2] NIPS 2017<br />[3] ICLR 2018 | ofirnachum |

### Others

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [lfads](lfads) | [LFADS - Latent Factor Analysis via Dynamical Systems](https://arxiv.org/abs/1608.06315) | | jazcollins, sussillo |
| [rebar](rebar) | [REBAR: Low-variance, unbiased gradient estimates for discrete latent variable models](https://arxiv.org/abs/1703.07370) | NIPS 2017 | gjtucker |

---

## :warning: Archived Models and Implementations

The following research models are no longer maintained.

**Note**: We will remove archived models from the master branch in June, 2020.
After removal, you will still be able to access archived models in the archive branch.

| Directory | Paper(s) | Conference | Maintainer(s) |
|-----------|----------|------------|---------------|
| [adv_imagenet_models](adv_imagenet_models) | [1] [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)<br />[2] [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) | [1] ICLR 2017<br /> [2] ICLR 2018 | alexeykurakin |
| [adversarial_crypto](adversarial_crypto) | [Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918) | | dave-andersen |
| [adversarial_logit_pairing](adversarial_logit_pairing) | [Adversarial Logit Pairing](https://arxiv.org/abs/1803.06373) | | alexeykurakin |
| [autoencoder](autoencoder) | Various autoencoders | | snurkabill |
| [brain_coder](brain_coder) | [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526) | | danabo, mnorouzi |
| [cognitive_mapping_and_planning](cognitive_mapping_and_planning) | [Cognitive Mapping and Planning for Visual Navigation](https://arxiv.org/abs/1702.03920) | CVPR 2017 | s-gupta |
| [compression](compression) | [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148) | CVPR 2017 | nmjohn |
| [deep_contextual_bandits](deep_contextual_bandits) | [Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127) | ICLR 2018 | rikel |
| [deep_speech](deep_speech) | [Deep Speech 2](https://arxiv.org/abs/1512.02595) | ICLR 2016 | yhliang2018 |
| [domain_adaptation](domain_adaptation) | [1] [Domain Separation Networks](https://arxiv.org/abs/1608.06019) <br />[2] [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1612.05424) | NIPS 2016 | bousmalis, dmrd |
| [feelvos](feelvos)| [FEELVOS](https://arxiv.org/abs/1902.09513) | CVPR 2019 | pvoigtlaender, yuningchai, aquariusjay |
| [fivo](fivo)| [Filtering variational objectives for training generative sequence models](https://arxiv.org/abs/1705.09279) | NIPS 2017 | dieterichlawson |
| [global_objectives](global_objectives) | [Scalable Learning of Non-Decomposable Objectives](https://arxiv.org/abs/1608.04802) | AISTATS 2017 | mackeya-google |
| [im2txt](im2txt) | [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge](https://arxiv.org/abs/1609.06647) | TPAMI 2016 | cshallue |
| [inception](inception) | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | CVPR 2016 | shlens, vincentvanhoucke |
| [keypointnet](keypointnet) | [KeypointNet](https://arxiv.org/abs/1807.03146) | | mnorouzi |
| [learned_optimizer](learned_optimizer) | [Learned Optimizers that Scale and Generalize](https://arxiv.org/abs/1703.04813) | ICML 2017 | olganw, nirum |
| [learning_to_remember_rare_events](learning_to_remember_rare_events) | [Learning to Remember Rare Events](https://arxiv.org/abs/1703.03129) | ICLR 2017| lukaszkaiser, ofirnachum |
| [learning_unsupervised_learning](learning_unsupervised_learning) | [Meta-Learning Update Rules for Unsupervised Representation Learning](https://arxiv.org/abs/1804.00222) | ICLR 2019 | lukemetz, nirum |
| [lexnet_nc](lexnet_nc) | [Olive Oil is Made of Olives, Baby Oil is Made for Babies: Interpreting Noun Compounds using Paraphrases in a Neural Model](https://arxiv.org/abs/1803.08073) | NAACL 2018 | vered1986, waterson |
| [lm_1b](lm_1b) | [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) | | oriolvinyals, panyx0718 |
| [lm_commonsense](lm_commonsense) | [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847) | | thtrieu |
| [maskgan](maskgan)| [MaskGAN: Better Text Generation via Filling in the](https://arxiv.org/abs/1801.07736) | ICLR 2018 | liamb315, a-dai |
| [namignizer](namignizer)| Namignizer | | knathanieltucker |
| [neural_gpu](neural_gpu)| [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) | | lukaszkaiser |
| [neural_programmer](neural_programmer) | [Learning a Natural Language Interface with Neural Programmer](https://arxiv.org/abs/1611.08945) | ICLR 2017 | arvind2505 |
| [next_frame_prediction](next_frame_prediction) | [Visual Dynamics: Probabilistic Future Frame Synthesis via Cross Convolutional Networks](https://arxiv.org/abs/1607.02586) | NIPS 2016 | panyx0718 |
| [ptn](ptn) | [Perspective Transformer Nets: Learning Single-View 3D Object Reconstruction without 3D Supervision](https://arxiv.org/abs/1612.00814) | NIPS 2016 | xcyan, arkanath, hellojas, honglaklee |
| [qa_kg](qa_kg) | [Learning to Reason: End-to-End Module Networks for Visual Question Answering](https://arxiv.org/abs/1704.05526) | ICCV 2017 | yuyuz |
| [real_nvp](real_nvp) | [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) | ICLR 2017 | laurent-dinh |
| [sentiment_analysis](sentiment_analysis)| [Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://arxiv.org/abs/1412.1058) | NAACL HLT 2015 | sculd |
| [seq2species](seq2species) | [Seq2Species: A deep learning approach to pattern recognition for short DNA sequences](https://doi.org/10.1101/353474) | | apbusia, depristo |
| [skip_thoughts](skip_thoughts) | [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726) | | cshallue |
| [steve](steve) | [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675) | NeurIPS 2018 | buckman-google |
| [street](street) | [End-to-End Interpretation of the French Street Name Signs Dataset](https://arxiv.org/abs/1702.03970) | ECCV 2016 | theraysmith |
| [struct2depth](struct2depth)| [Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/abs/1811.06152) | AAAI 2019 | aneliaangelova |
| [swivel](swivel) | [Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/abs/1602.02215) | | waterson |
| [tcn](tcn) | [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888) | ICRA 2018 | coreylynch, sermanet |
| [textsum](textsum)| [A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685) | EMNLP 2015 | panyx0718, peterjliu |
| [transformer](transformer) | [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) | NIPS 2015 | daviddao|
| [video_prediction](video_prediction) | [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157) | NIPS 2016 | cbfinn |

---

## Contributions

If you want to contribute, please review the [contribution guidelines](https://github.com/tensorflow/models/wiki/How-to-contribute).
