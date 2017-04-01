#!/bin/sh

cdir=$(pwd)

#specify arguments
trainzip=$cdir/dataset/dataset1/hw2_train.zip
testzip=$cdir/dataset/dataset1/hw2_test.zip
#1 for original word sequence, 2 for non-stop sequence
usewhich=2
hiddenlayersize=10
aerpha=0
maxiter=200
learnrate=0.01
momentu=0.4

trainzip2=$cdir/dataset/dataset2/enron1_train.zip
testzip2=$cdir/dataset/dataset2/enron1_test.zip

trainzip3=$cdir/dataset/dataset3/enron4_train.zip
testzip3=$cdir/dataset/dataset3/enron4_test.zip

cd $cdir/src
python testMLP.py $trainzip $testzip $usewhich $hiddenlayersize $aerpha $maxiter $learnrate $momentu
rm *.pyc