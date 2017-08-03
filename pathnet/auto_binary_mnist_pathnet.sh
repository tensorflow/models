#!/bin/bash

for i in {1..1000}
do
  python binary_mnist_pathnet.py > binary_mnist_pathnet.log
  result=`cat binary_mnist_pathnet.log |tail -1`
  echo $i "Iteration, " $result 
done
