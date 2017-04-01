'''
Created on Mar 15, 2017
Implement the perceptron algorithm
@author: mdy
'''

from zipMailParse import zipMailParse
import numpy as np

class Perceptron:
    def __init__(self, zip_path, use_which, learn_rate, iter_number):
        #using ham and spam to train
        #use_which = 1, use words sequence
        #use_which = 2, use non-stop words words sequence
        mpHam = zipMailParse(zip_path, 0)
        mpSpam = zipMailParse(zip_path, 1)
        self.hamtexts = []
        self.spamtexts = []
        
        self.usewhich = use_which
        #must specify one to use
        if use_which == 1:
            self.hamtexts = mpHam.D_wrd
            self.spamtexts = mpSpam.D_wrd
            
        if use_which == 2:
            self.hamtexts = mpHam.D_flt
            self.spamtexts = mpSpam.D_flt
            
        #0 for ham, and 1 for spam
        self.data_dict = {}
        self.data_dict.setdefault(0, self.hamtexts)
        self.data_dict.setdefault(1, self.spamtexts)
        
        #get vocabulary set
        self.voca_set = []
        self.get_voca_set()
        
        #get matrix for training
        self.xymat = self.get_xy_matrix()
        
        #initialize weight vector
        self.wvec = [0] * (len(self.voca_set) + 1)
        
        self.pcpTrain(learn_rate, iter_number)
            
        return
    
    def get_xy_matrix(self):
        xy_mat = []
        for key in self.data_dict.iterkeys():
            for doc in self.data_dict[key]:
                xvec = self.__getxvec(doc)
                xvec[-1] = key
                xy_mat.append(xvec)
                
        return xy_mat
    
    def __getxvec(self, doc):
        xvec = [0] * (len(self.voca_set) + 2)
        xvec[0] = 1
        for wd in doc:
            if wd in self.voca_set:
                ind = self.voca_set.index(wd)
                xvec[ind+1] += 1
                
        return xvec
    
    def get_voca_set(self):
        list1 = []
        for key in self.data_dict.iterkeys():
            list1.append(self.data_dict[key])
            
        list2 = []
        list2.extend(list1[0])
        list2.extend(list1[1])
        list3 = []
        for i in range(len(list2)):
            list3.extend(list2[i])
            
        self.voca_set = list(set(list3))
        return
    
    def pcpTrain(self, ita, iternum):
        cnt = 0
        while (cnt<iternum):
            for j in range(len(self.xymat)):
                tj = self.xymat[j][-1]
                oj = self.__sign_out(self.xymat[j])
                for k in range(len(self.wvec)):
                    delta_w_k = ita * (tj - oj) * self.xymat[j][k]
                    self.wvec[k] += delta_w_k
                    
            cnt += 1
            
        return
    
    def __sign_out(self, xyvec):
        x_vec = xyvec[:-1]
        w_vec = self.wvec
        dot_product_rst = np.dot(x_vec, w_vec)
        rst = 0
        if dot_product_rst>0:
            rst = 1
            
        return rst
    
    def applyPcp(self, t_path, hos):
        t_parse = zipMailParse(t_path, hos)
        if self.usewhich==1:
            tdoc = t_parse.D_wrd
            
        if self.usewhich==2:
            tdoc = t_parse.D_flt
            
        p_rst = []
        for i in range(len(tdoc)):
            #default prediction is ham
            p = 0
            xvec = self.__constructxvec(tdoc[i])
            wvec = self.wvec
            dotproduct = np.dot(xvec, wvec)
            if dotproduct>0:
                p=1
                
            p_rst.append(p)
            
        return p_rst
    
    def __constructxvec(self, doc):
        vec = [0] * (len(self.voca_set) + 1)
        vec[0] = 1
        for wd in doc:
            if wd in self.voca_set:
                ind = self.voca_set.index(wd)
                vec[ind + 1] += 1
                
        return vec
    
if __name__=='__main__':
    zip_path = '/home/mdy/Desktop/mlhw3/dataset/dataset2/enron1_train.zip'
    testpath = '/home/mdy/Desktop/mlhw3/dataset/dataset2/enron1_test.zip'
    pcp = Perceptron(zip_path, 1, 0.001, 19)
    print len(pcp.hamtexts)
    print len(pcp.spamtexts)
    print sum(pcp.wvec)
    prst = pcp.applyPcp(testpath, 0)
    print len(prst)
    print sum(prst)
    
    
    