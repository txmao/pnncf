'''
Created on Mar 16, 2017

multi-layer perceptron implementation

parameter considered:
 hidden_layer_sizes
 activation
 solver
 alpha
 batch_size
 learning_rate
 max_iter
 tol
 learning_rate_init
 momentum

@author: mdy
'''

from zipMailParse import zipMailParse
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import random
import copy

class MLPCimpl:
    #hidden layer size
    #l2 penalty
    #maximum iteration number
    #learning rate
    #momentum
    def __init__(self, trainzip, use_which, hiddenlayersize, aerpha, maxiteration, learnrate, momentu):
        #use_which=1, word sequence
        #use_which=2, non-stop word sequence
        self.usewhich = use_which
        
        wseqham = zipMailParse(trainzip, 0)
        wseqspam = zipMailParse(trainzip, 1)
        
        self.hamtexts = []
        self.spamtexts = []
        
        if use_which==1:
            self.hamtexts = wseqham.D_wrd
            self.spamtexts = wseqspam.D_wrd
            
        if use_which==2:
            self.hamtexts = wseqham.D_flt
            self.spamtexts = wseqspam.D_flt
            
        self.datadict = {}
        self.datadict.setdefault(0, self.hamtexts)
        self.datadict.setdefault(1, self.spamtexts)
        
        self.vocasetlist = []
        self.__get_vocasetlist()
        
        #X matrix and label y, original integer version
        self.XYinfo = []
        self.__getXYinfo()
        
        self.X_mat = []
        self.Y_vec = []
        self.__shuffleMAT()
        
        self.mlpcer = MLPClassifier(
            hidden_layer_sizes=hiddenlayersize,#
            activation='logistic',
            solver='sgd',
            alpha=aerpha,
            max_iter=maxiteration,#
            learning_rate_init=learnrate,#
            momentum=momentu,#
            batch_size=1
            )
        
        '''
        self.mlpcer = MLPClassifier(
            hidden_layer_sizes=hiddenlayersize
            )
        '''
        
        X = copy.deepcopy(self.X_mat)
        Y = copy.deepcopy(self.Y_vec)
        self.mlpcer.fit(X, Y)
        
        return
    
    def __shuffleMAT(self):
        ls1 = copy.deepcopy(self.XYinfo)
        ls2 = random.sample(ls1, len(ls1))
        for i in range(len(ls2)):
            xinfo = ls2[i][:-1]
            yinfo = ls2[i][-1]
            self.X_mat.append(xinfo)
            self.Y_vec.append(yinfo)
            
        return
    
    def __get_vocasetlist(self):
        list1 = []
        for k in self.datadict.iterkeys():
            list1.extend(self.datadict[k])
            
        list2 = []
        for i in range(len(list1)):
            list2.extend(list1[i])
            
        list3 = list(set(list2))
        self.vocasetlist = list3
        
    def __getXYinfo(self):
        XY = []
        for k in self.datadict.iterkeys():
            for doc in self.datadict[k]:
                xlist = self.__getXlist(doc)
                yval = k
                xylist = copy.deepcopy(xlist)
                xylist.append(yval)
                XY.append(xylist)
                
        self.XYinfo = copy.deepcopy(XY)
        return
    
    def __getXlist(self, doc):
        tmp = [0] * len(self.vocasetlist)
        for i in range(len(doc)):
            if doc[i] in self.vocasetlist:
                ind = self.vocasetlist.index(doc[i])
                tmp[ind] += 1
                
        return tmp
    
    def doXpreproc(self, Xmat):
        scaler = StandardScaler()
        scaler.fit(Xmat)
        X_s = scaler.transform(Xmat)
        return X_s
    
    def mclpPredict(self, tpath, hos):
        #hos=0, ham
        #hos=1, spam
        tparse = zipMailParse(tpath, hos)
        if self.usewhich==1:
            tdata = tparse.D_wrd
            
        if self.usewhich==2:
            tdata = tparse.D_flt
            
        tdataxmat = self.__extractXmat(tdata)
        #tdataxmat_after = self.doXpreproc(tdataxmat)
        tdataxmat_after = copy.deepcopy(tdataxmat)
        #predict
        rstarray = self.mlpcer.predict(tdataxmat_after)
        rst = list(rstarray)
        return rst
    
    def __extractXmat(self, tdata):
        xmat = []
        for doc in tdata:
            xvec = [0] * len(self.vocasetlist)
            for d in doc:
                if d in self.vocasetlist:
                    ind = self.vocasetlist.index(d)
                    xvec[ind] += 1
                    
            xmat.append(xvec)
            
        return xmat
    
if __name__=='__main__':
    hls = (5,)#
    acti = 'logistic'
    slvr = 'sgd'
    aerpha = 0
    batsize = 'auto'
    lrate = 'constant'
    mxiter = 10#
    tl = 1e-4
    lrateinit = 0.01#
    momen = 0.9#
    zp = '../dataset/dataset1/hw2_train.zip'
    zpt = '../dataset/dataset1/hw2_test.zip'
    mlpc = MLPCimpl(zp, 2, hls, aerpha, mxiter, lrateinit, momen)
    #print mlpc.X_mat_after[0]
    prst = mlpc.mclpPredict(zpt, 1)
    print len(prst)
    print sum(prst)
    
    
    