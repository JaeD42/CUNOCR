import os
import sys
import lib.usefulFunctions as use_func

"""
make sure we are on path
"""
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"

from Nets.Networks import SiameseNetMetric as mNet
from Nets.Networks import SiameseNetClassic as sNet
import tensorflow as tf

class NetCreator(object):
    """
    Helper class for creating Nets and loading/saving/training them
    """

    def __init__(self,creator_func=None,batch_size = 128, px = 48,p_same=0.05, encoding_size=600, conv_layer_size=[20, 40, 60, 100], conv_dim=[7, 5, 3, 3],fcl_layer_size=[], netClass = sNet):
        """
                Give lots of parameters to define the network completely
                :param creator_func: In case we have a function to create the net, should be deprecated. Simply keep it at None and nothing bad will happen
                :param batch_size: Recommended batch size for the network
                :param px: image width/height (images should all be square, so only one value)
                :param p_same: additional probability of drawing positive sample when doing random training as there are so few
                :param encoding_size: Size of the encoding vector
                :param conv_layer_size: list of integers describing the number of convolutions in each layer
                :param conv_dim: list of integers describing the convolution size in each layer (all convs are square sized)
                :param fcl_layer_size: list of integers describing the fcl size in each layer (can be empty, in this case there will only be a fully connected layer from the last convolution to the encoding vector and one from the encoding vector to output)
                :param netClass: Which networkt type should be used, metric or classic
        """

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


        self.saver = tf.train.Saver()


        self.trainDataLoader = None
        self.testDataLoader = None

    def addTrainDataLoader(self,LoaderClass, path):
        """
        Add a dataloader for training data
        :param LoaderClass: Type of loader
        :param path: path of data
        :return: the instanciated loader
        """
        self.trainDataLoader = LoaderClass(self.px, path)
        return self.trainDataLoader

    def addTestDataLoader(self,LoaderClass, path):
        """
        Add a dataloader for testing data
        :param LoaderClass: Type of loader
        :param path: path of data
        :return: the instanciated loader
        """
        self.testDataLoader = LoaderClass(self.px, path)
        return self.testDataLoader


    def saveNet(self, session, folder_name="", optional_file_name="", addendum=""):
        """
        Saves the network to file, automatically assumes project folder as source!
        :param session: session where the network was created in
        :param folder_name: Path to folder (based from project folder)
        :param optional_file_name: Name of file, if none given uses values provided in init to create a file name
        :param addendum: aditional string added to end of file to make filenames unique
        :return:
        """
        if folder_name=="":
            folder_name="savedNets"

        path = folder_path+folder_name
        use_func.createPath(path)

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
        """
        Load a network from file
        :param session: a tf.Session
        :param path: path to file
        :return:
        """
        self.saver.restore(session, path)


