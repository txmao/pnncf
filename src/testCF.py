'''
Created on Mar 19, 2017

@author: mdy
'''

from mbcf import mbcf
import sys

def testCF():
    print '--- collaborative filtering ---'
    zippath = str(sys.argv[1])
    psize = int(sys.argv[2])
    cf = mbcf(zippath, psize)
    print '~ prediction size: '+str(psize)
    print 'mean absolute error: '+str(cf.mae)
    print 'root mean square error: '+str(cf.rmse)
    return

if __name__=='__main__':
    testCF()