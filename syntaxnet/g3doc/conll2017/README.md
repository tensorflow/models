# CoNLL2017 Shared Task Instructions

We are pleased to provide a competitive baseline for the [CoNLL2017 Shared Task
on Dependency Parsing](http://universaldependencies.org/conll17/). Note that we
are providing detailed tutorials to make it easier to use DRAGNN as a platform
for improving upon the baselines.

## Running the baselines

*   Install SyntaxNet/DRAGNN following the install instructions in README.md
*   Download the models here: [link]
*   Download the contest [data data and
    tools](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-1976]).
*   Run the baseline_eval.py to run the pre-trained tokenizer and evaluate on
    the dev set.

You should obtain the following results on the dev sets:

NOTE: This will be filled in when the latest model results are available.

Language | No. tokens | Tokenization F1 | UAS | LAS
-------- | :--------: | :-------------: | :-: | :-:
Chinese  | XX         | XX              | XX  | XX

## Using DRAGNN for developing your own models

We hope that DRAGNN will be useful as a starting point for deep learning parsing
methods. We've provided a few recipes for alternative baselines in the examples/
directory; look for more coming soon!
