#!/bin/sh

cdir=$(pwd)

#arguments for training and testing perceptron
train_p=$cdir/dataset/dataset1/hw2_train.zip
test_p=$cdir/dataset/dataset1/hw2_test.zip
#1 for original word sequence, 2 for non-stop sequence
usewhich=2
ita=0.0006
iter=25

train_p2=$cdir/dataset/dataset2/enron1_train.zip
test_p2=$cdir/dataset/dataset2/enron1_test.zip

train_p3=$cdir/dataset/dataset3/enron4_train.zip
test_p3=$cdir/dataset/dataset3/enron4_test.zip

cd $cdir/src
python testPerceptron.py $train_p3 $test_p3 $usewhich $ita $iter
rm *.pyc