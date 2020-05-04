> :memo: A README.md template for releasing a paper code implementation to a GitHub repository.  
>  
> * Template version: 1.0.2020.125  
> * Provide general sections. (Please modify sections depending on needs.)  

# Model name, Paper title, or Project Name

This repository is the official or unofficial implementation of the following paper.

* Paper title: [Paper Title](https://arxiv.org/abs/YYMM.NNNNN)
* ArXiv identifier: [arXiv:YYMM.NNNNN](https://arxiv.org/abs/YYMM.NNNNN)

## Description

> :memo: Provide description of the model.  
>  
> * Provide brief information of the algorithms used.  
> * Provide links for demos, blog posts, etc.  

## History

> :memo: Provide a changelog.

## Maintainers

> :memo: Provide maintainer information.  

* Last name, First name ([@GitHub username](https://github.com/username))
* Last name, First name ([@GitHub username](https://github.com/username))

## Table of Contents

> :memo: Provide a table of contents to help readers navigate a lengthy README document.

## Requirements

> :memo: Provide details of the software required.  
>  
> * Add a `requirements.txt` file to the root directory for installing the necessary dependencies.  
>   * Describe how to install requirements using pip.  
> * Alternatively, create INSTALL.md.  

* TensorFlow requirement: 2.1

To install requirements:

```setup
pip install -r requirements.txt
```

## Results

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
