'''
Created on Mar 18, 2017

used to test the perceptron, two settings:
  learning rate
  number of iterations

<ham, spam>
dataset1: train<340, 123>, test<340, 130>
dataset2: train<319, 131>, test<307, 149>
dataset3: train<133, 402>, test<152, 391>

@author: mdy
'''

from __future__ import division
from Perceptron import Perceptron
import sys

def testPerceptron():
    print '------ Perceptron ------'
    #training set path
    train_p = str(sys.argv[1])
    #test set path
    test_p = str(sys.argv[2])
    #use word sequence 1, non-stop 2
    usewhich = int(sys.argv[3])
    #input learning rate
    ita = float(sys.argv[4])
    #input number of iterations
    iter = int(sys.argv[5])
    if usewhich==1:
        print 'use original word sequence'
        
    if usewhich==2:
        print 'use word sequence without stopwords'
        
    #train
    pcp = Perceptron(train_p, usewhich, ita, iter)
    
    #test
    #ham 0, or spam 1
    rst1 = pcp.applyPcp(test_p, 0)
    rst2 = pcp.applyPcp(test_p, 1)
    print len(rst1)
    print len(rst2)
    accr1 = ( (len(rst1) - sum(rst1)) / len(rst1) ) * 100
    accr2 = ( sum(rst2) / len(rst2) ) * 100
    accr3 = ( (len(rst1) - sum(rst1) + sum(rst2)) / (len(rst1) + len(rst2)) ) * 100
    print 'accuracy on ham: '+str(accr1)+'%'
    print 'accuracy on spam: '+str(accr2)+'%'
    print 'total accuracy: '+str(accr3)+'%'

if __name__=='__main__':
    testPerceptron()
    
    