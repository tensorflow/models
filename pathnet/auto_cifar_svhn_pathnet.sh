#!/bin/bash

for i in {1..100}
do
  python cifar_svhn_pathnet.py > cifar_svhn_pathnet.log
  result=`cat cifar_svhn_pathnet.log |tail -1`
  echo $i "Iteration, " $result 
done
