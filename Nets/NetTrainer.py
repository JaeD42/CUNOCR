import lib.usefulFunctions as u_func
from tqdm import tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x,**kwargs):
        return x
class NetTrainer(object):

    def __init__(self,net,session,train_data_loader,test_data_loader):
        self.net=net
        self.session=session
        self.train_data = train_data_loader
        self.test_data = test_data_loader

    def train_epoch(self,batchsize,language=0,lr=0.01):
        costSum=0
        for x1,x2,y in tqdm(self.train_data.iterate_epoch(batchsize,language=language),total=self.train_data.get_epoch_size()/batchsize):
            #y=[[i] for i in y]
            _,cost,reg_cost = self.session.run([self.net.optimizer,self.net.class_cost,self.net.reg_cost],feed_dict={
                self.net.x1:x1,
                self.net.x2:x2,
                self.net.y_true:y,
                self.net.lr:lr})
            costSum+=cost
        print "Cost of %s"%costSum



    def test_epoch(self,batchsize,language=0,epoch=0, save=False):
        FP=0
        TP=0
        FN=0
        TN=0
        all_preds = []
        all_trues = []


        for x1,x2,y in tqdm(self.test_data.iterate_epoch(batchsize,language=language),total=self.test_data.get_epoch_size()/batchsize):
            y_pred = self.session.run([self.net.y_pred],feed_dict={
                self.net.x1:x1,
                self.net.x2:x2})
            y_pred=y_pred[0]
            all_preds.extend(y_pred)
            all_trues.extend(y)
            errorMat = u_func.getErrorMat(y_pred,y)
            FP += errorMat["FP"]
            TP += errorMat["TP"]
            FN += errorMat["FN"]
            TN += errorMat["TN"]
        if save:
            with open("epoch%s"%epoch,mode='w') as f:
                f.write(str(all_preds))
                f.write(str(all_trues))
        return {"TP":TP, "TN":TN, "FP":FP, "FN":FN}



if __name__=="__main__":
    from Nets.NetCreator import NetCreator as NC
    import Nets.SiameseMetric as Siam
    import tensorflow as tf
    from lib.loader import OmniGlotLoader,CuneiformSetLoader
    from Nets.SiameseMetric import SiameseNetMetric as sNet

    folder = u_func.getFolderPath()+"Data/Datasets/"
    netCreator = NC(batch_size=128,px=48,netClass=sNet)

    sess=tf.Session()

    netCreator.initNet(sess)
    #netCreator.loadNet(sess,u_func.getFolderPath()+"savedNets/SiameseBackup4_2_fin2.ckpt")
    oLoader = OmniGlotLoader(48)
    netTrainer = NetTrainer(netCreator.net,sess, oLoader.training_loader, oLoader.testing_loader)
    for i in range(10):
        x=netTrainer.test_epoch(128, epoch=i,save=True)
        print x["TP"], x["TN"], x["FP"], x["FN"]
        x=netTrainer.train_epoch(128)


        netCreator.saveNet(sess,folder_name="fold6/",addendum="Test_%s"%i)



