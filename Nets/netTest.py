import tensorflow as tf
import Nets.SiameseNet as sNet
from lib import loader
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc


def load_net():
    sess = tf.Session()
    net, saver = sNet.runInit(sNet.backup4Net)
    sNet.runRestore(sess, saver, "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_2_fin2.ckpt")

    return net, sess


def get_loader(px=48):
    oLoad = loader.OmniGlotLoader(px)
    return oLoad

"""
oLoad = get_loader()
net,sess = load_net()
x1,x2,y = oLoad.get_training_sample(2,p_same=0)
enc1 = sess.run(net.enc1,feed_dict={net.x1:x1})

print enc1
"""
path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_Addition_fin2.ckpt"
#path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_2_fin.ckpt"
#path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseMetricBackup2_240000.ckpt"
import Nets.SiameseNet as sNet
#import Nets.SiameseMetric as sNet
import tensorflow as tf
import time
import Evaluation.Evals as evalFunc
import matplotlib.pyplot as plt
from lib.loader import OmniGlotLoader as OLoader
load = OLoader(48)
#net,saver = sNet.runInit(sNet.backup2Net)
net,saver = sNet.runInit(sNet.backup4Net)
print "starting calculations"

with tf.Session() as sess:
    sNet.runRestore(sess, saver, path)
    evalFunc.incresed_perf(net,sess,load)
    #pred,truth = evalFunc.test_perf(net,sess,load,0,2)