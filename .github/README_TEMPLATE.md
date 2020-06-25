> :memo: A README.md template for releasing a paper code implementation to a GitHub repository.  
>  
> * Template version: 1.0.2020.170  
> * Please modify sections depending on needs.  

# Model name, Paper title, or Project Name

> :memo: Add a badge for the ArXiv identifier of your paper (arXiv:YYMM.NNNNN)

[![Paper](http://img.shields.io/badge/Paper-arXiv.YYMM.NNNNN-B3181B?logo=arXiv)](https://arxiv.org/abs/...)

This repository is the official or unofficial implementation of the following paper.

* Paper title: [Paper Title](https://arxiv.org/abs/YYMM.NNNNN)

## Description

> :memo: Provide description of the model.  
>  
> * Provide brief information of the algorithms used.  
> * Provide links for demos, blog posts, etc.  

## History

> :memo: Provide a changelog.

## Authors or Maintainers

> :memo: Provide maintainer information.  

* Full name ([@GitHub username](https://github.com/username))
* Full name ([@GitHub username](https://github.com/username))

## Table of Contents

> :memo: Provide a table of contents to help readers navigate a lengthy README document.

## Requirements

[![TensorFlow 2.1](https://img.shields.io/badge/TensorFlow-2.1-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.1.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

> :memo: Provide details of the software required.  
>  
> * Add a `requirements.txt` file to the root directory for installing the necessary dependencies.  
>   * Describe how to install requirements using pip.  
> * Alternatively, create INSTALL.md.  

To install requirements:

```setup
pip install -r requirements.txt
```

## Results

[![TensorFlow Hub](https://img.shields.io/badge/TF%20Hub-Models-FF6F00?logo=tensorflow)](https://tfhub.dev/...)

> :memo: Provide a table with results. (e.g., accuracy, latency)  
>  
> * Provide links to the pre-trained models (checkpoint, SavedModel files).  
>   * Publish TensorFlow SavedModel files on TensorFlow Hub (tfhub.dev) if possible.  
> * Add links to [TensorBoard.dev](https://tensorboard.dev/) for visualizing metrics.  
>  
> An example table for image classification results  
>  
> ### Image Classification  
>  
> | Model name | Download | Top 1 Accuracy | Top 5 Accuracy |  
> |------------|----------|----------------|----------------|  
> | Model name | [Checkpoint](https://drive.google.com/...), [SavedModel](https://tfhub.dev/...) | xx% | xx% |  

## Dataset

> :memo: Provide information of the dataset used.  

## Training

> :memo: Provide training information.  
>  
> * Provide details for preprocessing, hyperparameters, random seeds, and environment.  
> * Provide a command line example for training.  

Please run this command line for training.

```shell
python3 ...
```

## Evaluation

> :memo: Provide an evaluation script with details of how to reproduce results.  
>  
> * Describe data preprocessing / postprocessing steps.  
> * Provide a command line example for evaluation.  

Please run this command line for evaluation.

```shell
python3 ...
```

## References

> :memo: Provide links to references.  

## License

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> :memo: Place your license text in a file named LICENSE in the root of the repository.  
>  
> * Include information about your license.  
> * Reference: [Adding a license to a repository](https://help.github.com/en/github/building-a-strong-community/adding-a-license-to-a-repository)  

This project is licensed under the terms of the **Apache License 2.0**.

## Citation

> :memo: Make your repository citable.  
>  
> * Reference: [Making Your Code Citable](https://guides.github.com/activities/citable-code/)  

If you want to cite this repository in your research paper, please use the following information.
