import os
import sys
import lib.usefulFunctions as u_func

from Nets.SiameseMetric import SiameseNetMetric as mNet
from Nets.SiameseNet import SiameseNetClassic as sNet
import tensorflow as tf

class NetCreator(object):
    """
    Helper class for creating Nets and loading/saving/training them
    """

    def __init__(self,creator_func=None,batch_size = 128, px = 48,p_same=0.05, encoding_size=600, conv_layer_size=[20, 40, 60, 100], conv_dim=[7, 5, 3, 3],fcl_layer_size=[], netClass = sNet):

        if creator_func!=None:
            self.batch_size, self.px, self.net, self.p_same = creator_func()
            self.creator_func=creator_func
        else:
            self.batch_size=batch_size
            self.px=px
            self.p_same=p_same
            self.net = netClass([None, px, px, 1], fcl_layer_size=fcl_layer_size, encoding_size=encoding_size,
                            conv_layer_size=conv_layer_size, conv_dim=conv_dim, regularization=True, reg_constant=0.01)
            self.enc=encoding_size
            self.clay = conv_layer_size
            self.cdim = conv_dim

        self.net.build()
        self.saver = tf.train.Saver()






    def saveNet(self, session, folder_name="", optional_file_name="", addendum=""):
        folder_path = u_func.getFolderPath()

        if folder_name=="":
            folder_name="savedNets"


        path = folder_path+folder_name
        u_func.createPath(path)

        if optional_file_name=="":
            if self.creator_func == None:
                optional_file_name = "%s_px%s_enc%s_clay%s_cdim%s"%(self.net.__class__.__name__,self.enc,self.clay,self.cdim)
            else:
                optional_file_name = "%s_func%s"%(self.net.__class__.__name__,self.creator_func.__name__)
        optional_file_name=path+optional_file_name+addendum
        num=0
        num_str=""
        while os.path.exists(optional_file_name+"_"+num_str+".ckpt"):
            num+=1
            num_str=str(num)



        save_path = self.saver.save(session, optional_file_name+num_str+".ckpt")

        print "Net saved in "+save_path

    def loadNet(self,session,path):
        self.saver.restore(session, path)


