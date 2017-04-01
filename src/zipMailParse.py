'''
Created on Mar 16, 2017

@author: mdy
'''

import os
import zipfile
import string

class zipMailParse:
    #0 for ham, 1 for spam
    def __init__(self, zp_path, ham_or_spam):
        #use the whole folder path
        self.D_raw = []
        self.D_wrd = []
        self.D_flt = []
        #original sequence, word sequence, non-stop word sequence
        self.stop_set = self.construct_stop_set()
        #stop words set
        self.parse_mail(zp_path, ham_or_spam)
        return
    
    def construct_stop_set(self):
        stpp = '../dataset/stopwords'
        with open(stpp) as f:
            ln = f.read().split()
            ln = [word.strip(string.punctuation).lower() for word in ln if word]
            stop_set = set(ln)
            
        return stop_set
    
    def parse_mail(self, zp_path, ham_or_spam):
        #choose ham or spam
        hos = ''
        if ham_or_spam==0:
            hos = 'ham'
            
        if ham_or_spam==1:
            hos = 'spam'
            
        zf = zipfile.ZipFile(zp_path)
        for fname in zf.namelist():
            if hos in fname and 'txt' in fname:
                with zf.open(fname) as f:
                    ln = f.read().split()
                    words1 = [wd for wd in ln]
                    #only consider lower case
                    words2 = [wd.lower() for wd in words1 if wd]
                    self.D_raw.append(words2)
                    words3 = [wd.strip(string.punctuation) for wd in ln]
                    words4 = [wd.lower() for wd in words3 if wd]
                    self.D_wrd.append(words4)
                    self.D_flt.append( self.__filtStop(self.stop_set, words4) )
                    
        zf.close()
        return
    
    def __filtStop(self, stpset, wds):
        rst = []
        for wd in wds:
            if wd not in stpset:
                rst.append(wd)
                
        return rst
    
if __name__=='__main__':
    zp = '/home/mdy/Desktop/mlhw3/dataset/dataset2/enron1_train.zip'
    zmp = zipMailParse(zp, 1)
    print len(zmp.D_raw)
    
    