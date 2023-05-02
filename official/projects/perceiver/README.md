# Perceiver IO: A General Architecture for Structured Inputs & Outputs

TF2 implementation of [Perceiver](https://arxiv.org/abs/2107.14795).

## Default setup command:
Scripts to pretrain, finetune, train from scratch can be found under
perceiver/experiments.

## BERT Wiki Books Pretrain

Configurations can be seen on Table 8 and Table 9 of the
 [paper](https://arxiv.org/abs/2107.14795). Our model configuration can be
  deduced in the configs and experiment folder, where we follow the
  configuration in the paper except for the tokenization and data.

Model | Tokenizer | Pretrain Data | Batch Size | Steps | Val MLM Accuracy
----- | --------: | ------------: | ---------: | ----: | ---------------:
Perceiver IO Base (paper) | SentencePiece | T5 + Wiki | 512 | 500 k | N/A
Perceiver IO Base (ours) | WordPiece | Wiki + Books | 512 | 500 k | 68.69 %

## GLUE Finetune

Our perceiver model is fine-tuned on GLUE upon the pre-trained model shown
 above. These are all single-task fine-tuning only.

These are run with configurations shown on Table 10 in the [paper](https://arxiv.org/abs/2107.14795).

Model | Tokenizer | Pretrain Data | CoLA | MNLI-m/mm | MRPC | QNLI | QQP | RTE | SST-2 | STS-B | Average
----- | --------: | ------------: | ---: | --------: | ----:| ----:| --: | --: | ----: | ----: | -----:
Perceiver IO Base (paper) | SentencePiece | T5 + Wiki | 47.11 % | 84.53/85.03 % | 87.25 % | 92.12 % | 90.22 % | 65.23 % | 94.38 % | 88.18 % | 81.16 %
Perceiver IO Base (ours) | WordPiece | Wiki + Books | 63.23 % | 84.29/84.52 % | 87.74 % | 91.43 % | 91.22 % | 70.76 % | 94.15 % | 89.85 % | 84.09 %

Note: The average is computed by first averaging the results of MNLI-matched and
MNLI-mismatched, which is then counted as a single task in the overall average.

`Average = (63.23 + (84.29 + 84.52) / 2 + 87.74 + 91.43 + 91.22 + 70.76 + 94.15 + 89.85) / 8`

## Discrepancy with the paper:

*   ~+2.93 average GLUE accuracy compared to paper results.

## Citing TensorFlow Model Garden

If you find this codebase helpful in your research, please cite this repository.

```
@misc{tensorflowmodelgarden2022,
  author = {Hongkun Yu and Chen Chen and Xianzhi Du and Yeqing Li and
            Abdullah Rashwan and Le Hou and Pengchong Jin and Fan Yang and
            Frederick Liu and Jaeyoun Kim and Jing Li},
  title = {{TensorFlow Model Garden}},
  howpublished = {\url{https://github.com/tensorflow/models}},
  year = {2020}
}
```