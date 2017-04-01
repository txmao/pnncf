'''
Created on Mar 19, 2017

test multilayer perceptron, parameter used:
  trainzip
  testzip
  usewhich
  hiddenlayersize #
  aerpha
  maxiter #
  learnrate #
  momentu #

@author: mdy
'''

from __future__ import division
from MLPCimpl import MLPCimpl
import sys
from __builtin__ import str

def testMLP():
    print '... multilayer perceptron ...'
    trainzip = str(sys.argv[1])
    testzip = str(sys.argv[2])
    #1 for word sequence, 2 for non stop
    usewhich = int(sys.argv[3])
    hiddenlayersize = int(sys.argv[4])
    aerpha = float(sys.argv[5])
    maxiter = int(sys.argv[6])
    learnrate = float(sys.argv[7])
    momentu = float(sys.argv[8])
    
    print 'hidden layer number: '+str(hiddenlayersize)
    print 'iteration number: '+str(maxiter)
    print 'learning rate: '+str(learnrate)
    print 'momentum: '+str(momentu)
    print '~~~'
    
    #train
    mpt = MLPCimpl(trainzip, usewhich, hiddenlayersize, aerpha, maxiter, learnrate, momentu)
    
    #test
    rst1 = mpt.mclpPredict(testzip, 0)
    rst2 = mpt.mclpPredict(testzip, 1)
    
    accr1 = ((len(rst1) - sum(rst1)) / len(rst1)) * 100
    accr2 = (sum(rst2) / len(rst2)) * 100
    accr3 = ((len(rst1) - sum(rst1) + sum(rst2)) / (len(rst1) + len(rst2))) * 100
    
    print 'accuracy on ham: '+str(accr1)+'%'
    print 'accuracy on spam: '+str(accr2)+'%'
    print 'total accuracy: '+str(accr3)+'%'
    
if __name__=='__main__':
    testMLP()