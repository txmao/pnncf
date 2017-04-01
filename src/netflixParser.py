'''
Created on Mar 16, 2017
used to parse the netflix.zip
movie_id and user_id in testing set is the subset of the training set 
@author: mdy
'''

import zipfile

class netflixParser:
    def __init__(self, nflixpath, pwhich):
        #pwhich=1, parse the Training
        #pwhich=2, parse the Testing
        #for training
        #MovieID, UserID, Rating
        self.movieuserrate_raw = []
        self.movie_raw = []
        self.user_raw = []
        self.rate_raw = []
        #parse
        self.parseWhich(nflixpath, pwhich)
        return
    
    def parseWhich(self, nflixpath, pwhich):
        pref = ''
        if pwhich==1:
            pref = 'Train'
            
        if pwhich==2:
            pref = 'Test'
            
        zf = zipfile.ZipFile(nflixpath)
        for fname in zf.namelist():
            if pref in fname:
                with zf.open(fname) as f:
                    lns = f.readlines()
                    lns1 = [ln.strip().split(',') for ln in lns]
                    #note only one such file
                    #self.movieuserrate_raw = lns1
                    for i in range(len(lns1)):
                        mid = int(lns1[i][0])
                        uid = int(lns1[i][1])
                        frate = float(lns1[i][2])
                        mur_new = [mid, uid, frate]
                        self.movieuserrate_raw.append(mur_new)
                        self.movie_raw.append(mid)
                        self.user_raw.append(uid)
                        self.rate_raw.append(frate)
                    
        zf.close()
        return
    
if __name__=='__main__':
    nflzipp = '../dataset/netflix/netflix.zip'
    pwh = 1
    nfp = netflixParser(nflzipp, pwh)
    print len(nfp.movieuserrate_raw)
    print nfp.movieuserrate_raw[3255351]
    
    
    