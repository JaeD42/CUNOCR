import tensorflow as tf
import Nets.SiameseNet as sNet
from lib import loader
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from collections import Counter
import lib.usefulFunctions as u_func
class netTest(object):
    class minClassEnc(object):
        def __init__(self,enc,ind):
            self.enc=enc
            self.ind=ind

    def __init__(self,loader,net,sess):
        self.data = loader.dataset[0]
        self.net = net
        self.session = sess
        self.encClass = [self.minClassEnc(u_func.getEncoding([i],net,sess),ind)for ind,vals in enumerate(self.data) for i in vals ]
        self.encs = [i.enc for i in self.encClass]
        print "Finished"

    def testKNN(self,k):
        self.dist = self.net.calc_dist_mat_from_encs(self.session,self.encs,batchsize=128)
        correct=0
        incorrect=0
        for ind,row in enumerate(self.dist):
            row_w_num = zip(row,[i.ind for i in self.encClass[ind]])
            trueClass = row_w_num[ind][1]
            row_w_num.sort(key=lambda x:x[0])
            classes = [i[1] for i in row_w_num[:k]]
            counted = Counter(classes)
            counted[trueClass]-=1
            predClass = counted.most_common(n=1)[0][0]

            correct+=(predClass==trueClass)
            incorrect+=(predClass!=trueClass)

        return correct,incorrect





