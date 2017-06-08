# Parsey Universal.

A collection of pretrained syntactic models is now available for download at
`http://download.tensorflow.org/models/parsey_universal/<language>.zip`

After downloading and unzipping a model, you can run it similarly to
Parsey McParseface with:

```shell
  MODEL_DIRECTORY=/where/you/unzipped/the/model/files
  cat sentences.txt | syntaxnet/models/parsey_universal/parse.sh \
    $MODEL_DIRECTORY > output.conll
```

These models are trained on
[Universal Dependencies](http://universaldependencies.org/) datasets v1.3.
The following table shows their accuracy on Universal
Dependencies test sets for different types of annotations.

Language | No. tokens | POS | fPOS | Morph | UAS | LAS
--------  | :--: | :--: | :--: | :--: | :--: | :--:
Ancient_Greek-PROIEL | 18502 | 97.14% | 96.97% | 89.77% | 78.74% | 73.15%
Ancient_Greek | 25251 | 93.22% | 84.22% | 90.01% | 68.98% | 62.07%
Arabic | 28268 | 95.65% | 91.03% | 91.23% | 81.49% | 75.82%
Basque | 24374 | 94.88% | - | 87.82% | 78.00% | 73.36%
Bulgarian | 15734 | 97.71% | 95.14% | 94.61% | 89.35% | 85.01%
Catalan | 59503 | 98.06% | 98.06% | 97.56% | 90.47% | 87.64%
Chinese | 12012 | 91.32% | 90.89% | 98.76% | 76.71% | 71.24%
Croatian | 4125 | 94.67% | - | 86.69% | 80.65% | 74.06%
Czech-CAC | 10862 | 98.11% | 92.43% | 91.43% | 87.28% | 83.44%
Czech-CLTT | 4105 | 95.79% | 87.36% | 86.33% | 77.34% | 73.40%
Czech | 173920 | 98.12% | 93.76% | 93.13% | 89.47% | 85.93%
Danish | 5884 | 95.28% | - | 95.24% | 79.84% | 76.34%
Dutch-LassySmall | 4562 | 95.62% | - | 95.44% | 81.63% | 78.08%
Dutch | 5843 | 89.89% | 86.03% | 89.12% | 77.70% | 71.21%
English-LinES | 8481 | 95.34% | 93.11% | - | 81.50% | 77.37%
English | 25096 | 90.48% | 89.71% | 91.30% | 84.79% | 80.38%
Estonian | 23670 | 95.92% | 96.76% | 92.73% | 83.10% | 78.83%
Finnish-FTB | 16286 | 93.50% | 91.15% | 92.44% | 84.97% | 80.48%
Finnish | 9140 | 94.78% | 95.84% | 92.42% | 83.65% | 79.60%
French | 7018 | 96.27% | - | 96.05% | 84.68% | 81.05%
Galician | 29746 | 96.81% | 96.14% | - | 84.48% | 81.35%
German | 16268 | 91.79% | - | - | 79.73% | 74.07%
Gothic | 5158 | 95.58% | 96.03% | 87.32% | 79.33% | 71.69%
Greek | 5668 | 97.48% | 97.48% | 92.70% | 83.68% | 79.99%
Hebrew | 12125 | 95.04% | 95.04% | 92.05% | 84.61% | 78.71%
Hindi | 35430 | 96.45% | 95.77% | 90.98% | 93.04% | 89.32%
Hungarian | 4235 | 94.00% | - | 75.68% | 78.75% | 71.83%
Indonesian | 11780 | 92.62% | - | - | 80.03% | 72.99%
Irish | 3821 | 91.34% | 89.95% | 77.07% | 74.51% | 66.29%
Italian | 10952 | 97.31% | 97.18% | 97.27% | 89.81% | 87.13%
Kazakh | 587 | 75.47% | 75.13% | - | 58.09% | 43.95%
Latin-ITTB | 6548 | 97.98% | 92.68% | 93.52% | 84.22% | 81.17%
Latin-PROIEL | 14906 | 96.50% | 96.08% | 88.39% | 77.60% | 70.98%
Latin | 4832 | 88.04% | 74.07% | 76.03% | 56.00% | 45.80%
Latvian | 3985 | 80.95% | 66.60% | 73.60% | 58.92% | 51.47%
Norwegian | 30034 | 97.44% | - | 95.58% | 88.61% | 86.22%
Old_Church_Slavonic | 5079 | 96.50% | 96.28% | 89.43% | 84.86% | 78.85%
Persian | 16022 | 96.20% | 95.72% | 95.90% | 84.42% | 80.28%
Polish | 7185 | 95.05% | 85.83% | 86.12% | 88.30% | 82.71%
Portuguese-BR | 29438 | 97.07% | 97.07% | 99.91% | 87.91% | 85.44%
Portuguese | 6262 | 96.81% | 90.67% | 94.22% | 85.12% | 81.28%
Romanian | 18375 | 95.26% | 91.66% | 91.98% | 83.64% | 75.36%
Russian-SynTagRus | 107737 | 98.27% | - | 94.91% | 91.68% | 87.44%
Russian | 9573 | 95.27% | 95.02% | 87.75% | 81.75% | 77.71%
Slovenian-SST | 2951 | 90.00% | 84.48% | 84.38% | 65.06% | 56.96%
Slovenian | 14063 | 96.22% | 90.46% | 90.35% | 87.71% | 84.60%
Spanish-AnCora | 53594 | 98.28% | 98.28% | 97.82% | 89.26% | 86.50%
Spanish | 7953 | 95.27% | - | 95.74% | 85.06% | 81.53%
Swedish-LinES | 8228 | 96.00% | 93.77% | - | 81.38% | 77.21%
Swedish | 20377 | 96.27% | 94.13% | 94.14% | 83.84% | 80.28%
Tamil | 1989 | 79.29% | 71.79% | 75.97% | 64.45% | 55.35%
Turkish | 8616 | 93.63% | 92.62% | 86.79% | 82.00% | 71.37%
**Average** | - | 94.27% | 92.93% | 90.38% | 81.12% | 75.85%

These results are obtained using gold text segmentation. Accuracies are measured
over all tokens, including punctuation. `POS`, `fPOS` are coarse and fine
part-of-speech tagging accuracies. `Morph` is full-token accuracy of predicted
morphological attributes. `UAS` and `LAS` are unlabeled and labeled attachment
scores.

Many of these models also support text segmentation, with:

```shell
  MODEL_DIRECTORY=/where/you/unzipped/the/model/files
  cat sentences.txt | syntaxnet/models/parsey_universal/tokenize.sh \
    $MODEL_DIRECTORY > output.conll
```

Text segmentation is currently available for:
`Bulgarian`, `Czech`, `German`, `Greek`, `English`, `Spanish`, `Estonian`,
`Basque`, `Persian`, `Finnish`, `Finnish-FTB`, `French`, `Galician`,
`Ancient_Greek`, `Ancient_Greek-PROIEL`, `Hebrew`, `Hindi`, `Croatian`,
`Hungarian`, `Indonesian`, `Italian`, `Latin`, `Latin-PROIEL`, `Dutch`,
`Norwegian`, `Polish`, `Portuguese`, `Slovenian`, `Swedish`, `Tamil`.

For `Chinese` (traditional) we use a larger text segmentation
model, which can be run with:

```shell
  MODEL_DIRECTORY=/where/you/unzipped/the/model/files
  cat sentences.txt | syntaxnet/models/parsey_universal/tokenize_zh.sh \
    $MODEL_DIRECTORY > output.conll
```
