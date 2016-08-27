training tagger and parser
===

### download univeral dependency data( http://universaldependencies.org/#en )
```shell
$ cd ud_train
$ mkdir corpus
$ cd corpus
(downloading ud-treebanks-*.tgz)
$ tar -zxvf ud-treebanks-*.tgz  
$ ls universal-dependencies-* 
$ UD_Ancient_Greek  UD_Basque  UD_Czech ....
```

### train
```shell
# for example, training UD_English
$ cp -rf corpus/UD_English/*.conllu UD/
# if you use other corpus, set proper corpus path in UD/context.pbtxt

$ ./train.sh -v -v
...
#preprocessing with tagger
INFO:tensorflow:Seconds elapsed in evaluation: 9.77, eval metric: 99.71%
INFO:tensorflow:Seconds elapsed in evaluation: 1.26, eval metric: 92.04%
INFO:tensorflow:Seconds elapsed in evaluation: 1.26, eval metric: 92.07%
...
#pretrain parser
INFO:tensorflow:Seconds elapsed in evaluation: 4.97, eval metric: 82.20%
...
#evaluate pretrained parser
INFO:tensorflow:Seconds elapsed in evaluation: 44.30, eval metric: 92.36%
INFO:tensorflow:Seconds elapsed in evaluation: 5.42, eval metric: 82.67%
INFO:tensorflow:Seconds elapsed in evaluation: 5.59, eval metric: 82.36%
...
#train parser
INFO:tensorflow:Seconds elapsed in evaluation: 57.69, eval metric: 83.95%
...
#evaluate parser
INFO:tensorflow:Seconds elapsed in evaluation: 283.77, eval metric: 96.54%
INFO:tensorflow:Seconds elapsed in evaluation: 34.49, eval metric: 84.09%
INFO:tensorflow:Seconds elapsed in evaluation: 34.97, eval metric: 83.49%
...
```

### test
```shell
$ echo "this is my own tagger and parser" | ./test.sh
...
Input: this is my own tagger and parser
Parse:
tagger NN ROOT
 +-- this DT nsubj
 +-- is VBZ cop
 +-- my PRP$ nmod:poss
 +-- own JJ amod
 +-- and CC cc
 +-- parser NN conj
```

