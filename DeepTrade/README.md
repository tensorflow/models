A LSTM model using Risk Estimation loss function for trades in market
===

## Introduction

   Could deep learning help us with buying and selling stocks in market? The answer could be 'Yes'. We design a solution, named DeepTrade, including history data representation, neural network construction and trading optimization methods, which could maximizing our profit based on passed experience.

   In our solution, effective representations are extracted from history data (including date/open/high/low/close/volume) first. Then a neural network based on LSTM is constructed to learn useful knowledges to direct our trading behaviors. Meanwhile, a loss function is elaborately designed to ensure the network optimizing our profit and minimizing our risk. Finaly, according the predictions of this neural network, buying and selling plans are carried out.

## Feature Representation

   History features are extracted in the order of date. Each day, with open/high/low/close/volume data, invariant features are computed, including rate of price change, MACD, RSI, rate of volume change, BOLL, distance between MA and price, distance between volume MA and volume, cross feature between price and volume. Some of these features could be used directly. Some of them should be normalized. And some should use diffrential values. A fixed length(i.e., 30 days) of feature is extracted for network learning.

## Network Construction

   LSTM network [1] is effective with learning knowleges from time series. A fixed length of history data (i.e., 30 days) is used to plan trade of next day. We make the network output a real value (p) between 0 and 1, which means how much position (in percent) of the stock we should hold to tomorrow. So that if the rate of price change is r next day, out profit will be p*r. If r is negtive, we lost our money. Therefore, we define a Loss Function (called Risk Estimation) for the LSTM network:

   Loss = -100. * mean(P * R)

P is a set of our output, and R is the set of corresponding rates of price change. Further more, we add a small cost rate (c=0.0002) for money occupied by buying stock to the loss function. Then the loss function with cost rate is defined as follows:
   
   Loss = -100. * mean(P * (R - c))

  Both of these two loss functions are evaluated in our experiments.

  Our network includes four layers: LSTM layer, dense connected layer, batch normalization [3] layer, activation layer. LSTM layer is used to learn knowldges from histories. The relu6 function is used as activation to produce output value.  

## Trading Plans

   Every day, at the time before market close (nearer is better), input history features into the network, then we get an output value p. This p mean an advice of next-day's position. If p=0, we should sell all we have before close. If p is positive, we should keep a poistion of p to next day, sell the redundant or buy the insufficient.

## Experimental Results

   If the network goes crazy(overfitting), just restart it. Or, a dropout layer [2] is good idea. Also, larger train dataset will help.
 
   For more demos of the experimental results, visit our website: http://www.deeplearning.xin.
   
   [Experimental Results](http://www.deeplearning.xin)
   
## Requirements

ta-lib, ta-lib for python, numpy, tensorflow

## Liecence

The author is Xiaoyu Fang from China. Please quot the source whenever you use it.

## Bug Report

Contact happynoom@163.com to report any bugs.

## Reference

[1] Gers F A, Schmidhuber J, Cummins F, et al. Learning to Forget: Continual Prediction with LSTM[J]. Neural Computation, 2000, 12(10): 2451-2471.

[2] Srivastava N, Hinton G E, Krizhevsky A, et al. Dropout: a simple way to prevent neural networks from overfitting[J]. Journal of Machine Learning Research, 2014, 15(1): 1929-1958.

[3] Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift[C]. international conference on machine learning, 2015: 448-456.



