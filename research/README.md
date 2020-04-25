![Logo](https://storage.googleapis.com/model_garden_artifacts/TF_Model_Garden.png)

# TensorFlow Research Models

This folder contains machine learning models implemented by researchers in [TensorFlow](https://tensorflow.org). 

The research models are maintained by their respective authors. 

**Note: Some research models are stale and have not updated to the latest TensorFlow 2 yet.**

---

## Frameworks / APIs with Models
| Folder | Framework | Description | Maintainer(s) |
|--------|-----------|-------------|---------------|
| [object_detection](object_detection) | TensorFlow Object Detection API | A framework that makes it easy to construct, train and deploy object detection models<br/> | jch1, tombstone, derekjchow, jesu9, dreamdragon, pkulzc |
| [slim](slim) | TensorFlow-Slim Image Classification Model Library | A lightweight high-level API of TensorFlow for defining, training and evaluating image classification models <br/>• Inception V1/V2/V3/V4<br/>• Inception-ResNet-v2<br/>• ResNet V1/V2<br/>• VGG 16/19<br/>• MobileNet V1/V2/V3<br/>• NASNet-A_Mobile/Large<br/>• PNASNet-5_Large/Mobile | sguada, nathansilberman |

---

## Models / Implementations

| Folder | Paper(s) | Description | Maintainer(s) |
|--------|----------|-------------|---------------|
| [adv_imagenet<br />_models](adv_imagenet_models)   | [1] [Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)<br/>[2] [Ensemble Adversarial Training: Attacks and Defenses](https://arxiv.org/abs/1705.07204) | Adversarially trained ImageNet models  | alexeykurakin  |
| [adversarial_crypto](adversarial_crypto) | [Learning to Protect Communications with Adversarial Neural Cryptography](https://arxiv.org/abs/1610.06918) | Code to train encoder/decoder/adversary network triplets and evaluate their effectiveness on randomly generated input and key pairs | dave-andersen |
| [adversarial<br />_logit_pairing](adversarial_logit_pairing)   | [Adversarial Logit Pairing](https://arxiv.org/abs/1803.06373) | Implementation of Adversarial logit pairing paper as well as few models pre-trained on ImageNet and Tiny ImageNet   | alexeykurakin |
| [adversarial_text](adversarial_text) | [1] [Adversarial Training Methods for Semi-Supervised Text](https://arxiv.org/abs/1605.07725) Classification<br/>[2] [Semi-supervised Sequence Learning](https://arxiv.org/abs/1511.01432) | Adversarial Training Methods for Semi-Supervised Text Classification| rsepassi, a-dai |
| [attention_ocr](attention_ocr)   | [Attention-based Extraction of Structured Information from Street View Imagery](https://arxiv.org/abs/1704.03549) | | xavigibert |
| [audioset](audioset) | Models for AudioSet: A Large Scale Dataset of Audio Events | | plakal, dpwe |
| [autoaugment](autoaugment) | [1] [AutoAugment](https://arxiv.org/abs/1805.09501)<br/>[2] [Wide Residual Networks](https://arxiv.org/abs/1605.07146)<br/>[3] [Shake-Shake regularization](https://arxiv.org/abs/1705.07485)<br/>[4] [ShakeDrop Regularization for Deep Residual Learning](https://arxiv.org/abs/1802.02375) | Train Wide-ResNet, Shake-Shake and ShakeDrop models on CIFAR-10 and CIFAR-100 dataset with AutoAugment | barretzoph |
| [autoencoder](autoencoder) | Various autoencoders | | snurkabill |
| [brain_coder](brain_coder) | [Neural Program Synthesis with Priority Queue Training](https://arxiv.org/abs/1801.03526) | Program synthesis with reinforcement learning  | danabo |
| [cognitive_mapping<br />_and_planning](cognitive_mapping_and_planning) | [Cognitive Mapping and Planning for Visual Navigation](https://arxiv.org/abs/1702.03920) | Implementation of a spatial memory based mapping and planning architecture for visual navigation | s-gupta |
| [compression](compression) | [Full Resolution Image Compression with Recurrent Neural Networks](https://arxiv.org/abs/1608.05148) | | nmjohn |
| [cvt_text](cvt_text) | [Semi-supervised sequence learning with cross-view training](https://arxiv.org/abs/1809.08370) | | clarkkev, lmthang |
| [deep_contextual<br />_bandits](deep_contextual_bandits) | [Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling](https://arxiv.org/abs/1802.09127) | | rikel |
| [deep_speech](deep_speech) | [Deep Speech 2](https://arxiv.org/abs/1512.02595) | End-to-End Speech Recognition in English and Mandarin | |
| [deeplab](deeplab)  | [1] [DeepLabv1](https://arxiv.org/abs/1412.7062)<br/>[2] [DeepLabv2](https://arxiv.org/abs/1606.00915)<br/>[3] [DeepLabv3](https://arxiv.org/abs/1802.02611)<br/>[4] [DeepLabv3+](https://arxiv.org/abs/1706.05587) | DeepLab models for semantic image segmentation | aquariusjay, yknzhu, gpapan |
| [delf](delf)  | [1] [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321) <br/>[2] [Detect-to-Retrieve](https://arxiv.org/abs/1812.01584) | DELF: DEep Local Features | andrefaraujo |
| [domain_adaptation](domain_adaptation) | [1] [Domain Separation Networks](https://arxiv.org/abs/1608.06019) <br/>[2] [Unsupervised Pixel-Level Domain Adaptation with Generative Adversarial Networks](https://arxiv.org/abs/1612.05424) | Code used for two domain adaptation papers| bousmalis, dmrd |
| [efficient-hrl](efficient-hrl) | [1] [Data-Efficient Hierarchical Reinforcement Learning](https://arxiv.org/abs/1805.08296)<br/>[2] [Near-Optimal Representation Learning for Hierarchical Reinforcement Learning](https://arxiv.org/abs/1810.01257) | Code for performing hierarchical reinforcement learning | ofirnachum |
| [feelvos](feelvos)| [FEELVOS](https://arxiv.org/abs/1902.09513) | Fast End-to-End Embedding Learning for Video Object Segmentation | |
| [fivo](fivo)| [Filtering variational objectives for training generative sequence models](https://arxiv.org/abs/1705.09279) | | dieterichlawson |
| [global_objectives](global_objectives) | [Scalable Learning of Non-Decomposable Objectives](https://arxiv.org/abs/1608.04802) | TensorFlow loss functions that optimize directly for a variety of objectives including AUC, recall at precision, and more | mackeya-google |
| [im2txt](im2txt) | [Show and Tell: Lessons learned from the 2015 MSCOCO Image Captioning Challenge](https://arxiv.org/abs/1609.06647) | Image-to-text neural network for image captioning| cshallue |
| [inception](inception) | [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567) | Deep convolutional networks for computer vision | shlens, vincentvanhoucke |
| [keypointnet](keypointnet) | [KeypointNet](https://arxiv.org/abs/1807.03146) | Discovery of Latent 3D Keypoints via End-to-end Geometric Reasoning | mnorouzi |
| [learned_optimizer](learned_optimizer) | [Learned Optimizers that Scale and Generalize](https://arxiv.org/abs/1703.04813) | | olganw, nirum |
| [learning_to<br />_remember<br />_rare_events](learning_to_remember_rare_events) | [Learning to Remember Rare Events](https://arxiv.org/abs/1703.03129) | A large-scale life-long memory module for use in deep learning | lukaszkaiser, ofirnachum |
| [learning<br />_unsupervised<br />_learning](learning_unsupervised_learning) | [Meta-Learning Update Rules for Unsupervised Representation Learning](https://arxiv.org/abs/1804.00222) | A meta-learned unsupervised learning update rule| lukemetz, nirum |
| [lexnet_nc](lexnet_nc) | LexNET | Noun Compound Relation Classification | vered1986, waterson |
| [lfads](lfads) | [LFADS - Latent Factor Analysis via Dynamical Systems](https://doi.org/10.1101/152884) | Sequential variational autoencoder for analyzing neuroscience data| jazcollins, sussillo |
| [lm_1b](lm_1b) | [Exploring the Limits of Language Modeling](https://arxiv.org/abs/1602.02410) | Language modeling on the one billion word benchmark | oriolvinyals, panyx0718 |
| [lm_commonsense](lm_commonsense) | [A Simple Method for Commonsense Reasoning](https://arxiv.org/abs/1806.02847) | Commonsense reasoning using language models | thtrieu |
| [lstm_object_detection](lstm_object_detection) | [Mobile Video Object Detection with Temporally-Aware Feature Maps](https://arxiv.org/abs/1711.06368) | | dreamdragon, masonliuw, yinxiaoli, yongzhe2160 |
| [marco](marco) | [Classification of crystallization outcomes using deep convolutional neural networks](https://arxiv.org/abs/1803.10342) | | vincentvanhoucke |
| [maskgan](maskgan)| [MaskGAN: Better Text Generation via Filling in the______](https://arxiv.org/abs/1801.07736) | Text generation with GANs | liamb315, a-dai |
| [namignizer](namignizer)| Namignizer | Recognize and generate names | knathanieltucker |
| [neural_gpu](neural_gpu)| [Neural GPUs Learn Algorithms](https://arxiv.org/abs/1511.08228) | Highly parallel neural computer | lukaszkaiser |
| [neural_programmer](neural_programmer) | [Learning a Natural Language Interface with Neural Programmer](https://arxiv.org/abs/1611.08945) | Neural network augmented with logic and mathematic operations| arvind2505 |
| [next_frame<br />_prediction](next_frame_prediction) | [Visual Dynamics](https://arxiv.org/abs/1607.02586) | Probabilistic Future Frame Synthesis via Cross Convolutional Networks| panyx0718 |
| [pcl_rl](pcl_rl) | [1] [Improving Policy Gradient by Exploring Under-appreciated Rewards](https://arxiv.org/abs/1611.09321)<br/>[2] [Bridging the Gap Between Value and Policy Based Reinforcement Learning](https://arxiv.org/abs/1702.08892)<br/>[3] [Trust-PCL: An Off-Policy Trust Region Method for Continuous Control](https://arxiv.org/abs/1707.01891) | Code for several reinforcement learning algorithms | ofirnachum |
| [ptn](ptn) | [Perspective Transformer Nets](https://arxiv.org/abs/1612.00814) | Learning Single-View 3D Object Reconstruction without 3D Supervision | xcyan, arkanath, hellojas, honglaklee |
| [qa_kg](qa_kg) | [Learning to Reason](https://arxiv.org/abs/1704.05526) | End-to-End Module Networks for Visual Question Answering | yuyuz |
| [real_nvp](real_nvp) | [Density estimation using Real NVP](https://arxiv.org/abs/1605.08803) | | laurent-dinh |
| [rebar](rebar) | [REBAR](https://arxiv.org/abs/1703.07370) | Low-variance, unbiased gradient estimates for discrete latent variable models | gjtucker |
| [sentiment<br />_analysis](sentiment_analysis)| [Effective Use of Word Order for Text Categorization with Convolutional Neural Networks](https://arxiv.org/abs/1412.1058) | A simple model to classify a document's sentiment | sculd |
| [seq2species](seq2species) | [Seq2Species: A deep learning approach to pattern recognition for short DNA sequences](https://doi.org/10.1101/353474) | Neural Network Models for Species Classification| apbusia, depristo |
| [skip_thoughts](skip_thoughts) | [Skip-Thought Vectors](https://arxiv.org/abs/1506.06726) | Recurrent neural network sentence-to-vector encoder | cshallue|
| [steve](steve) | [Sample-Efficient Reinforcement Learning with Stochastic Ensemble Value Expansion](https://arxiv.org/abs/1807.01675) | A hybrid model-based/model-free reinforcement learning algorithm for sample-efficient continuous control | buckman-google |
| [street](street) | [End-to-End Interpretation of the French Street Name Signs Dataset](https://arxiv.org/abs/1702.03970) | Identify the name of a street (in France) from an image using a Deep RNN| theraysmith |
| [struct2depth](struct2depth)| [Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos](https://arxiv.org/abs/1811.06152) | Unsupervised learning of depth and ego-motion| aneliaangelova |
| [swivel](swivel) | [Swivel: Improving Embeddings by Noticing What's Missing](https://arxiv.org/abs/1602.02215) | The Swivel algorithm for generating word embeddings | waterson |
| [tcn](tcn) | [Time-Contrastive Networks: Self-Supervised Learning from Video](https://arxiv.org/abs/1704.06888) | Self-supervised representation learning from multi-view video | coreylynch, sermanet |
| [textsum](textsum)| Sequence-to-sequence with attention model for text summarization | | panyx0718, peterjliu |
| [transformer](transformer) | [Spatial Transformer Network](https://arxiv.org/abs/1506.02025) | Spatial transformer network that allows the spatial manipulation of data within the network| daviddao|
| [vid2depth](vid2depth) | [Unsupervised Learning of Depth and Ego-Motion from Monocular Video Using 3D Geometric Constraints](https://arxiv.org/abs/1802.05522) | Learning depth and ego-motion unsupervised from raw monocular video | rezama |
| [video<br />_prediction](video_prediction) | [Unsupervised Learning for Physical Interaction through Video Prediction](https://arxiv.org/abs/1605.07157) | Predicting future video frames with neural advection| cbfinn |

---

## Contributions

If you want to contribute a new model, please submit a pull request.
