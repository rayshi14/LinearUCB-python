#!/usr/bin/env python2.7

import numpy as np
import math
from scipy import linalg


# lin UCB
class LinUCB:
    def __init__(self):
        # upper bound coefficient
        self.alpha = 3 # if worse -> 2.9, 2.8 1 + np.sqrt(np.log(2/delta)/2)
        self.r1 = 0.5 # if worse -> 0.7, 0.8
        self.r0 = -20 # if worse, -19, -21
        # dimension of user features = d
        self.d = 6
        # Aa : collection of matrix to compute disjoint part for each article a, d*d
        self.Aa = {}
        # AaI : store the inverse of all Aa matrix
        self.AaI = {}
        # ba : collection of vectors to compute disjoin part, d*1
        self.ba = {}
        
        self.a_max = 0
        
        self.theta = {}
        self.x = None
        self.xT = None
        # linUCB

    def set_articles(self, art):
        # init collection of matrix/vector Aa, Ba, ba
        for key in art:
            self.Aa[key] = np.identity(self.d)
            self.ba[key] = np.zeros((self.d, 1))
            self.AaI[key] = np.identity(self.d)
            self.theta[key] = np.zeros((self.d, 1))
            
    def update(self, reward):
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            if reward == 1:
                r = self.r1
            else:
                r = self.r0
            self.Aa[self.a_max] += self.x.dot(self.xT)
            self.ba[self.a_max] += r * self.x
            self.AaI[self.a_max] = linalg.solve(self.Aa[self.a_max], np.identity(self.d))
            self.theta[self.a_max] = self.AaI[self.a_max].dot(self.ba[self.a_max])
        else:
        # error
            pass
    
    def reccomend(self, timestamp, user_features, articles):
        xaT = np.array([user_features])
        xa = np.transpose(xaT)
        #art_max = -1
        #old_pa = 0
        pa = np.array([float(np.dot(xaT, self.theta[article]) + self.alpha * np.sqrt(np.dot(xaT.dot(self.AaI[article]), xa))) for article in articles])
        self.a_max = articles[divmod(pa.argmax(), pa.shape[0])[1]]
        '''
        for article in articles:
            # x : feature of current article, d*1
            # theta = self.AaI[article].dot(self.ba[article])
            sa = np.dot(xaT.dot(self.AaI[article]), xa)
            new_pa = float(np.dot(xaT, self.theta[article]) + self.alpha * np.sqrt(np.dot(xaT.dot(self.AaI[article]), xa)))
            if art_max == -1:
                old_pa = new_pa
                art_max = article
            else:
                if old_pa < new_pa:
                    art_max = article
                    old_pa = new_pa
        '''    
        self.x = xa
        self.xT = xaT
        # article index with largest UCB
        #self.a_max = art_max # divmod(pa.argmax(), pa.shape[0])[1]
        
        return self.a_max  
    
# UCB2
class UCB2:
    def __init__(self):

        # upper bound coefficient
        self.alpha = 0.1
        
        self.article_features = {}
        
        self.a_max = 0
        self.epoch = 0
        self.n = 0

    # Evaluator will call this function and pass the article features.
    # Check evaluator.py description for details.
    def set_articles(self, art):
        for key in art:
            # tuple (xj, nj, rj)
            self.article_features[key] = (0, 0, 0)

    # This function will be called by the evaluator.
    # Check task description for details.
    def update(self, reward):
        if self.epoch > 0: # still in the middle of an epoch
            self.epoch -= 1
        if reward == -1:
            pass
        elif reward == 1 or reward == 0:
            # update
            # first time play
            if self.article_features[self.a_max][1] == 0:
                self.article_features[self.a_max] = (reward, 1, 1)
            else: # not the first time, check epoch
                nj = self.article_features[self.a_max][1] + 1
                xj = (self.article_features[self.a_max][0]*self.article_features[self.a_max][1] + reward)/nj
                if self.epoch > 0: # still in the middle of an epoch
                    rj = self.article_features[self.a_max][2]
                else: # epoch finished
                    rj = self.article_features[self.a_max][2] + 1
                self.article_features[self.a_max] = (xj, nj, rj)
        else:
        # error
            pass

    # This function will be called by the evaluator.
    # Check task description for details.
    def reccomend(self, timestamp, user_features, articles):
    
        for article in articles:
            if self.article_features[article][1] == 0:
                self.n += 1
                self.a_max = article
                return self.a_max
        
        if self.epoch == 0:
            old_a = -1
            old_ucb = 0
            
            for article in articles:
                xj = self.article_features[article][0]
                rj = self.article_features[article][2]
                nj = self.article_features[article][1]
                tr = math.ceil(math.pow((1+self.alpha), rj))
                anr = math.sqrt((1+self.alpha)*math.log(math.e*self.n/tr)/(2*tr))
                new_ucb =  xj + anr
                if old_a == -1:
                    old_ucb = new_ucb
                    old_a = article
                else:
                    if new_ucb > old_ucb:
                        old_a = article
                        old_ucb = new_ucb
            
            rj = self.article_features[old_a][2]
            tr1 = math.ceil(math.pow((1+self.alpha), rj+1))
            tr = math.ceil(math.pow((1+self.alpha), rj))
            self.epoch = max(1, tr1 - tr)
            self.a_max = old_a
        
        self.n += 1
    
        return self.a_max


LinUCBObj = None
#UCB2Obj = None
#t = 0

def set_articles(art):
    global LinUCBObj#, UCB2Obj
    LinUCBObj = LinUCB()
    LinUCBObj.set_articles(art)
    #UCB2Obj = UCB2()
    #UCB2Obj.set_articles(art)

def update(reward):
    #global t
    #if t > 5000000:
    #    t += 1
    #    return UCB2Obj.update(reward)
    #else:
    #t += 1
    return LinUCBObj.update(reward)   

# This function will be called by the evaluator.
# Check task description for details.
def reccomend(timestamp, user_features, articles):
    #if t > 5000000:
    #    return UCB2Obj.reccomend(timestamp, user_features, articles)
    #else:
    return LinUCBObj.reccomend(timestamp, user_features, articles)
