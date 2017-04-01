'''
Created on Mar 16, 2017
memory based collaborative filtering
@author: mdy
'''

from __future__ import division
from netflixParser import netflixParser
import copy
import numpy as np
import time
import random

class mbcf:
    def __init__(self, zippath, p_size):
        #extract the training data
        #use 1 for training, 2 for testing
        nfp_train = netflixParser(zippath, 1)
        #4 raw data
        self.murraw = nfp_train.movieuserrate_raw
        self.mraw = nfp_train.movie_raw
        self.uraw = nfp_train.user_raw
        self.rraw = nfp_train.rate_raw
        #set list
        self.msetlist = list(set(self.mraw))
        self.usetlist = list(set(self.uraw))
        #map id to index
        self.movie_id_index = {}
        self.user_id_index = {}
        self.__id_to_index()
        #initial vote matrix, vij, ith user's vote for jth's movie
        #Vmatini = np.zeros([len(self.usetlist), len(self.msetlist)])
        self.Vmat_ini = [[0.0]*len(self.msetlist) for i in range(len(self.usetlist))]
        self.Vi_average = [0] * len(self.usetlist)
        self.__getVmat_ini()
        #vij - average(vi)
        self.Vmat = copy.deepcopy(self.Vmat_ini)
        #reduce size
        self.Vmat_ini = []
        self.__getVmat()
        #kwai, very big mat
        #self.kwai = [[0]*len(self.usetlist) for i in range(len(self.usetlist))]
        #self.__getkwai()
        ###
        ###predictiong part, use size control, 0 without control, 
        ###
        nfp_test = netflixParser(zippath, 2) #2 for testing
        self.murraw_test = nfp_test.movieuserrate_raw
        self.rraw_test = nfp_test.rate_raw
        self.psize = p_size
        self.prstlist = []
        #mean absolute error, root mean square error
        self.mae = 0
        self.rmse = 0
        self.usedsample = []
        if p_size==0:
            self.predictALL()
            pass
        else:
            self.predictSIZE(p_size)
            pass
        
        self.rraw_test_sample = []
        self.getcorrectedtestrate()
        self.geterrorinfo()
        return
    
    def getcorrectedtestrate(self):
        for ite in self.usedsample:
            self.rraw_test_sample.append(self.rraw_test[ite])
            
        return
    
    def geterrorinfo(self):
        lenp = len(self.prstlist)
        subvec = np.subtract(self.prstlist, self.rraw_test_sample)
        sum1 = 0
        sum2 = 0
        for i in range(len(subvec)):
            sum1 += abs(subvec[i])
            sum2 += (subvec[i]*subvec[i])
            
        self.mae = sum1/lenp
        self.rmse = np.sqrt(sum2/lenp)
        return
    
    def __id_to_index(self):
        #a very time consuming process
        #movie id to index
        for i in range(len(self.msetlist)):
            self.movie_id_index.setdefault(self.msetlist[i], i)
            
        #user id to index
        for j in range(len(self.usetlist)):
            self.user_id_index.setdefault(self.usetlist[j], j)
            
        return
    
    def __getVmat_ini(self):
        #update all
        for i in range(len(self.murraw)):
            s_mid = self.movie_id_index.get(self.mraw[i])
            s_uid = self.user_id_index.get(self.uraw[i])
            s_rate = self.rraw[i]
            self.Vmat_ini[s_uid][s_mid] += s_rate
            self.Vi_average[s_uid] += s_rate
            
        for j in range(len(self.Vi_average)):
            deno = len(self.msetlist) - self.Vmat_ini[j].count(0)
            self.Vi_average[j] = self.Vi_average[j]/deno
            
        return
    
    def __getVmat(self):
        for i in range(len(self.usetlist)):
            for j in range(len(self.msetlist)):
                #self.Vmat[i][j] += -(self.Vi_average[i])
                if self.Vmat[i][j] != 0:
                    self.Vmat[i][j] += -(self.Vi_average[i])
                    
        return
    
    def __getkwai(self):
        for a in range(len(self.usetlist)):
            for i in range(len(self.usetlist)):
                if i>=a:
                    f1 = np.dot(self.Vmat[a], self.Vmat[i])
                    f2 = np.dot(self.Vmat[a], self.Vmat[a])
                    f3 = np.dot(self.Vmat[i], self.Vmat[i])
                    f4 = (f1)/( np.sqrt(f2*f3) )
                    self.kwai[a][i] = f4
                    self.kwai[i][a] = f4
                    
        #do for k
        for k in range(len(self.kwai)):
            sum_k = sum(self.kwai[k])
            for j in range(len(self.kwai[k])):
                self.kwai[k][j] = self.kwai[k][j] / sum_k
                
        return
    
    def predictALL(self):
        for i in range(len(self.murraw_test)):
            user_a_ind = self.user_id_index.get(self.murraw_test[i][1])
            movie_j_ind = self.movie_id_index.get(self.murraw_test[i][0])
            vij_sub_avevi = []
            for u in range(len(self.Vmat)):
                vij_sub_avevi.append(self.Vmat[u][movie_j_ind])
                
            kwai = []
            for ii in range(len(self.usetlist)):
                f1 = np.dot(self.Vmat[user_a_ind], self.Vmat[ii])
                f2 = np.dot(self.Vmat[user_a_ind], self.Vmat[user_a_ind])
                f3 = np.dot(self.Vmat[ii], self.Vmat[ii])
                if f2==0 or f3==0:
                    f4 = 0
                else:
                    f4 = (f1) / (np.sqrt(f2*f3))
                        
                kwai.append(f4)
                
            sum_kk = sum([abs(ki) for ki in kwai])
            for kk in range(len(kwai)):
                if sum_kk!=0:
                    kwai[kk] = kwai[kk] / sum_kk
                else:
                    kwai[kk] = 0
                
            #do ith prediction
            ave_a = self.Vi_average[user_a_ind]
            paj = ave_a + np.dot(kwai, vij_sub_avevi)
            self.prstlist.append(paj)
            
        return
    
    def predictSIZE(self, psize):
        t1 = time.time()
        rdmdt = range(0, len(self.murraw_test))
        rdmdtsp = random.sample(rdmdt, psize)
        self.usedsample = copy.deepcopy(rdmdtsp)
        for i in rdmdtsp:
            if i>=0:
                user_a_ind = self.user_id_index.get(self.murraw_test[i][1])
                movie_j_ind = self.movie_id_index.get(self.murraw_test[i][0])
                vij_sub_avevi = []
                for u in range(len(self.Vmat)):
                    vij_sub_avevi.append(self.Vmat[u][movie_j_ind])
                    
                kwai = []
                for ii in range(len(self.usetlist)):
                    f1 = np.dot(self.Vmat[user_a_ind], self.Vmat[ii])
                    f2 = np.dot(self.Vmat[user_a_ind], self.Vmat[user_a_ind])
                    f3 = np.dot(self.Vmat[ii], self.Vmat[ii])
                    if f2==0 or f3==0:
                        f4 = 0
                    else:
                        f4 = (f1) / (np.sqrt(f2*f3))
                        
                    kwai.append(f4)
                
                #sum_kk = sum(kwai)
                
                sum_kk = sum([abs(ki) for ki in kwai])
                for kk in range(len(kwai)):
                    if sum_kk!=0:
                        kwai[kk] = kwai[kk] / sum_kk
                    else:
                        kwai[kk] = 0
                    
                #do ith prediction
                ave_a = self.Vi_average[user_a_ind]
                paj = ave_a + np.dot(kwai, vij_sub_avevi)
                self.prstlist.append(paj)
            
        t2 = time.time()
        print 'time: '+str(t2-t1)
        return
    
if __name__=='__main__':
    zppt = '../dataset/netflix/netflix.zip'
    sz = 1
    cf = mbcf(zppt, sz)
    #print cf.prstlist
    #print cf.rraw_test[:sz]
    print cf.mae
    print cf.rmse
    
    
    
    
