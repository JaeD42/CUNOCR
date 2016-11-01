import lib.usefulFunctions as u_func

class NetTrainer(object):

    def __init__(self,net,session,train_data_loader,test_data_loader):
        self.net=net
        self.sesseion=session
        self.train_data = train_data_loader
        self.test_data = test_data_loader

    def train_epoch(self,batchsize,language=0,lr=0.01):
        for x1,x2,y in self.train_data.iterate_epoch(batchsize,language=language):
            _,cost,_,_,reg_cost = self.session.run([self.optimizer,self.class_cost,self.enc1,self.enc2,self.reg_cost],feed_dict={
                self.x1:x1,
                self.x2:x2,
                self.y_true:y,
                self.lr:lr})



    def test_epoch(self,batchsize,language=0):
        FP=0
        TP=0
        FN=0
        TN=0
        for x1,x2,y in self.test_data.iterate_epoch(batchsize,language=language):
            y_pred = self.session.run([self.y_pred],feed_dict={
                self.x1:x1,
                self.x2:x2})
            errorMat = u_func.getErrorMat(y_pred,y)
            FP += errorMat["FP"]
            TP += errorMat["TP"]
            FN += errorMat["FN"]
            TN += errorMat["TN"]
        return {"TP":TP, "TN":TN, "FP":FP, "FN":FN}


