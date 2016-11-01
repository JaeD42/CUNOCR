"""
path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_2_fin2.ckpt"
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
s=time.time()
with tf.Session() as sess:
    sNet.runRestore(sess, saver, path)
    evalFunc.incresed_perf(net,sess,load)
"""