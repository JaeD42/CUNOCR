import lib.usefulFunctions as u_func

class NetTrainer(object):

    def __init__(self,net,session,train_data_loader,test_data_loader):
        self.net=net
        self.session=session
        self.train_data = train_data_loader
        self.test_data = test_data_loader

    def train_epoch(self,batchsize,language=0,lr=0.01):
        costSum=0
        for x1,x2,y in self.train_data.iterate_epoch(batchsize,language=language):
            #y=[[i] for i in y]
            _,cost,reg_cost = self.session.run([self.net.optimizer,self.net.class_cost,self.net.reg_cost],feed_dict={
                self.net.x1:x1,
                self.net.x2:x2,
                self.net.y_true:y,
                self.net.lr:lr})
            costSum+=cost
        print "Cost of %s"%costSum



    def test_epoch(self,batchsize,language=0,epoch=0):
        FP=0
        TP=0
        FN=0
        TN=0
        all_preds = []
        all_trues = []
        for x1,x2,y in self.test_data.iterate_epoch(batchsize,language=language):
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
        if epoch%10==0:
            with open("epoch%s"%epoch,mode='w') as f:
                f.write(str(all_preds))
                f.write(str(all_trues))
        return {"TP":TP, "TN":TN, "FP":FP, "FN":FN}



if __name__=="__main__":
    from Nets.NetCreator import NetCreator as NC
    import Nets.SiameseNet as Siam
    import tensorflow as tf
    from lib.loader import CuneiformSetLoader
    from Nets.SiameseNet import SiameseNetClassic as sNet

    folder = u_func.getFolderPath()+"Data/Datasets/"
    netCreator = NC(creator_func=Siam.backup4Net)

    sess=tf.Session()

    #netCreator.initNet(sess)
    netCreator.loadNet(sess,u_func.getFolderPath()+"savedNets/SiameseBackup4_2_fin2.ckpt")
    netTrainer = NetTrainer(netCreator.net,sess, CuneiformSetLoader(48,folder+"CleanFin"), CuneiformSetLoader(48,folder+"Dataset_Cuneiform"))
    for i in range(100):
        x=netTrainer.train_epoch(128)
        x=netTrainer.test_epoch(128,epoch=i)
        print x["TP"],x["TN"],x["FP"],x["FN"]
        netCreator.saveNet(sess,folder_name="fold3/",addendum="Test_%s"%i)



