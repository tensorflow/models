# CoNLL2017 Shared Task Instructions

We are pleased to provide a competitive baseline for the [CoNLL2017 Shared Task
on Dependency Parsing](http://universaldependencies.org/conll17/). Note that we
are providing detailed tutorials to make it easier to use DRAGNN as a platform
for improving upon the baselines.

Please see our [paper](paper.pdf) for more technical details about the model.

## Running the baselines

*   Install SyntaxNet/DRAGNN following the install instructions.
*   Download the models
    [here](https://drive.google.com/file/d/0BxpbZGYVZsEeSFdrUnBNMUp1YzQ/view?usp=sharing)
*   Download the contest [data and
    tools](http://universaldependencies.org/conll17/)
*   Run the baseline_eval.py to run the pre-trained tokenizer and evaluate on
    the dev set.

You should obtain the following results on the dev sets with gold segmentation.
Note: Our segmenter does not split multi-word tokens, which may not play nice
(yet) with the official evaluation script.

Language             | UAS   | LAS
-------------------- | :---: | :---:
Ancient_Greek-PROIEL | 81.52 | 76.87
Ancient_Greek        | 70.96 | 65.13
Arabic               | 84.79 | 78.90
Basque               | 80.96 | 77.19
Bulgarian            | 91.33 | 86.77
Catalan              | 91.32 | 88.76
Chinese              | 77.56 | 71.96
Croatian             | 86.62 | 81.84
Czech-CAC            | 89.99 | 86.09
Czech-CLTT           | 78.25 | 73.70
Czech                | 89.55 | 85.23
Danish               | 84.69 | 81.36
Dutch-LassySmall     | 84.12 | 80.85
Dutch                | 86.68 | 81.91
English-LinES        | 82.43 | 78.46
English-ParTUT       | 83.55 | 79.00
English              | 87.60 | 84.20
Estonian             | 75.77 | 67.76
Finnish-FTB          | 87.54 | 83.70
Finnish              | 87.05 | 83.33
French-ParTUT        | 85.12 | 80.79
French-Sequoia       | 87.90 | 85.74
French               | 91.05 | 88.48
Galician-TreeGal     | 75.26 | 69.50
Galician             | 84.64 | 81.58
German               | 85.53 | 81.27
Gothic               | 81.79 | 74.99
Greek                | 86.99 | 84.23
Hebrew               | 87.79 | 82.18
Hindi                | 93.73 | 90.10
Hungarian            | 78.68 | 73.03
Indonesian           | 83.02 | 76.51
Irish                | 75.02 | 65.66
Italian-ParTUT       | 85.09 | 80.90
Italian              | 90.73 | 87.71
Japanese             | 95.33 | 93.99
Kazakh               | 28.09 | 7.87
Korean               | 81.21 | 76.78
Latin-ITTB           | 82.86 | 78.43
Latin-PROIEL         | 79.52 | 73.58
Latin                | 64.72 | 54.59
Latvian              | 76.17 | 70.55
Norwegian-Bokmaal    | 91.23 | 88.79
Norwegian-Nynorsk    | 89.32 | 86.67
Old_Church_Slavonic  | 84.96 | 79.65
Persian              | 87.70 | 83.98
Polish               | 91.32 | 86.83
Portuguese-BR        | 92.36 | 90.60
Portuguese           | 90.60 | 88.12
Romanian             | 89.41 | 83.00
Russian-SynTagRus    | 91.51 | 89.05
Russian              | 85.18 | 80.71
Slovak               | 88.08 | 82.64
Slovenian-SST        | 66.77 | 59.38
Slovenian            | 89.85 | 87.62
Spanish-AnCora       | 91.02 | 88.61
Spanish              | 90.32 | 87.16
Swedish-LinES        | 83.67 | 78.96
Swedish              | 82.45 | 78.75
Turkish              | 68.81 | 60.57
Ukrainian            | 72.19 | 62.79
Urdu                 | 85.50 | 79.19
Uyghur               | 69.23 | 43.27
Vietnamese           | 65.18 | 55.61

## Using DRAGNN for developing your own models

We hope that DRAGNN will be useful as a starting point for deep learning parsing
methods. We've provided a few recipes for alternative baselines sprinkled
through the tutorials and examples.
