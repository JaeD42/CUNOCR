import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib.usefulFunctions as u_func

"""
Use tqdm if installed for nice percentages
"""
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x


class NetTrainer(object):
    """
    Container class for easier training of networks
    """
    def __init__(self, net, session, train_data_loader, test_data_loader):
        """
        Initialize net trainer with a network, a session and two dataloaders
        for training and testing
        :param net: A network
        :param session: A tf.Session
        :param train_data_loader: data loader for training
        :param test_data_loader: data loader for testing
        """
        self.net = net
        self.session = session
        self.train_data = train_data_loader
        self.test_data = test_data_loader

    def train_epoch(self, batchsize, language=-1, lr=0.01):
        """
        Train a single epoch on train_data,
        Loads every possible pair of characters (from the same language) and
        trains on them. For Omniglot a specific language can be chosen, otherwise
        all are used. Prints costs after training on epoch.
        :param batchsize: preferred batchsize
        :param language: specify a language, in case you want a specific one, otherwise all languages will be used
        :param lr: learning rate
        :return:
        """
        costSum = 0
        for x1, x2, y in tqdm(self.train_data.iterate_epoch(batchsize, language=language),
                              total=self.train_data.get_epoch_size() / batchsize):
            _, cost, reg_cost = self.session.run([self.net.optimizer, self.net.class_cost, self.net.reg_cost],
                                                 feed_dict={
                                                     self.net.x1: x1,
                                                     self.net.x2: x2,
                                                     self.net.y_true: y,
                                                     self.net.lr: lr})
            costSum += cost
        print "Cost of %s" % costSum

    def test_epoch(self, batchsize, language=-1, lim=0.5, epoch=0,save=False):
        """
        Test on the complete testing set.
        Loads every possible pair and predicts distance
        Returns a dictionary containing an error matrix
        A file containing all predicitons and truth values can be saved
        (CAREFUL: File saved can be very big when using complete Omniglot)
        :param batchsize: preferred batch size
        :param language: language to use for testing
        :param epoch: epoch we are in, used for saving files
        :param lim: limit for deciding if two images depict the same character. Influences the error matrix but not saving
        :param save: if predictions and truth values should be saved to file
        :return: Error matrix of the testing set
        """
        FP = 0
        TP = 0
        FN = 0
        TN = 0
        all_preds = []
        all_trues = []

        for x1, x2, y in tqdm(self.test_data.iterate_epoch(batchsize, language=language),
                              total=self.test_data.get_epoch_size() / batchsize):
            y_pred = self.session.run([self.net.y_pred], feed_dict={
                self.net.x1: x1,
                self.net.x2: x2})

            all_preds.extend(y_pred)
            all_trues.extend(y)
            errorMat = u_func.getErrorMat(y_pred, y, lim=lim)
            FP += errorMat["FP"]
            TP += errorMat["TP"]
            FN += errorMat["FN"]
            TN += errorMat["TN"]
        if save:
            with open("epoch%s" % epoch, mode='w') as f:
                f.write(str(all_preds))
                f.write(str(all_trues))
        return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


if __name__ == "__main__":
    from Nets.NetCreator import NetCreator as NC
    import Nets.SiameseMetric as Siam
    import tensorflow as tf
    from lib.DataSetLoader import OmniGlotLoader, CuneiformSetLoader
    from Nets.SiameseMetric import SiameseNetMetric as sNet
    from Nets.SiameseNet import SiameseNetClassic as cNet
    from time import time

    folder = u_func.getFolderPath() + "Data/Datasets/"

    netCreator = NC(batch_size=128, px=48, netClass=cNet)

    sess = tf.Session()

    # netCreator.initNet(sess)
    netCreator.loadNet(sess, u_func.getFolderPath() + "foldClassic/epoch5.ckpt")
    # oLoader = OmniGlotLoader(48)
    training = CuneiformSetLoader(48, u_func.getFolderPath() + "TrainingSet")
    testing = CuneiformSetLoader(48, u_func.getFolderPath() + "TestingSet")
    netTrainer = NetTrainer(netCreator.net, sess, training, testing)
    s = time()
    for i in range(25):
        print "Iteration took %s seconds" % (time() - s)
        x = netTrainer.test_epoch(128, epoch=i, save=True)
        print x["TP"], x["TN"], x["FP"], x["FN"]
        s = time()
        x = netTrainer.train_epoch(128)

        netCreator.saveNet(sess, folder_name="foldClassicAddTrain/", addendum="Test_%s" % i)



