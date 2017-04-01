#!/bin/sh

cdir=$(pwd)

#zip path and psize
zippath=$cdir/dataset/netflix/netflix.zip
#sample size
psize=1800

cd $cdir/src
python testCF.py $zippath $psize
rm *.pyc