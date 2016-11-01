import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"
import tensorflow as tf
from NetClass import Net
import ops
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x,**kwargs):
        return x


import numpy as np
class SiameseNetMetric(Net):

    def __init__(self,shapeImg,conv_layer_size = [20, 40, 60, 100], conv_dim = [7, 5, 3, 3], fcl_layer_size = [], encoding_size=400,lr=0.01, regularization=False,reg_constant=0.001):
        self.shape = shapeImg
        self.learning_rate=lr
        self.conv_layer_size = conv_layer_size
        self.conv_dim = conv_dim
        self.fcl_layer_size = fcl_layer_size
        self.fcl_layer_size.append(encoding_size)
        self.encoding_size = encoding_size
        self.enc_weights=[]
        self.regularization =regularization
        self.reg_constant = reg_constant



    def build_encoding(self,x):
        def getVars(name,w_shape):
            w=tf.get_variable("W"+name,w_shape,initializer=tf.random_normal_initializer(mean=0.0,stddev=0.1),regularizer=tf.contrib.layers.l2_regularizer(0.01))
            b = tf.get_variable("b"+name,[w_shape[-1]],initializer=tf.constant_initializer(0.1),regularizer=tf.contrib.layers.l2_regularizer(0.01))
            return w,b
        prev_layer=x
        img_size=self.shape[2]
        for ind in range(len(self.conv_layer_size)):
            if ind == 0:
                w_shape = [self.conv_dim[ind], self.conv_dim[ind], self.shape[3], self.conv_layer_size[ind]]
            else:
                w_shape = [self.conv_dim[ind], self.conv_dim[ind], self.conv_layer_size[ind - 1], self.conv_layer_size[ind]]
            w,b = getVars("enc%s"%ind,w_shape)
            prev_layer = ops.max_pool_2x2(tf.nn.relu(ops.conv_2d(prev_layer, w,b)))

            self.enc_weights.append(w)
            self.enc_weights.append(b)
        next_size=self.conv_layer_size[-1]*img_size/(2**len(self.conv_layer_size))*img_size/(2**len(self.conv_layer_size))
        flat_layer = tf.reshape(prev_layer, [-1, next_size])

        for ind in range(len(self.fcl_layer_size)):
            if ind == 0:
                w_shape = [next_size,self.fcl_layer_size[0]]
            else:
                w_shape = [self.fcl_layer_size[ind-1],self.fcl_layer_size[ind]]
            w,b = getVars("enc_fcl%s"%ind,w_shape)
            flat_layer = tf.nn.sigmoid(tf.matmul(flat_layer,w) + b)
            self.enc_weights.append(w)
            self.enc_weights.append(b)

        return flat_layer



    def build(self,**inputArgs):

        self.lr = tf.placeholder(tf.float32, shape=[])
        self.x1 = tf.placeholder(tf.float32, shape=self.shape, name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=self.shape, name="x2")

        self.y_true = tf.placeholder(tf.float32, shape=[self.shape[0], 1])

        with tf.variable_scope("image_filters") as scope:
            self.enc1=self.build_encoding(self.x1)
            scope.reuse_variables()
            self.enc2=self.build_encoding(self.x2)

        self.params = self.enc_weights
        self.dec_weights = ops.conv_weight_bias([self.encoding_size,1])

        self.params.extend(self.dec_weights)
        self.y_pred = tf.nn.tanh(tf.matmul(tf.abs(self.enc1-self.enc2), tf.square(self.dec_weights[0])))

        self.class_cost = tf.reduce_mean(
            -tf.mul(1 - self.y_true, tf.log(tf.clip_by_value(self.y_pred, 0.0001, 0.9999)))) - tf.reduce_mean(
            tf.mul(self.y_true, tf.log(0.99991 - tf.clip_by_value(self.y_pred, 0.0001, 0.9999))))

        self.cost = self.class_cost
        self.reg_cost = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*self.reg_constant


        if self.regularization:
            self.cost+=self.reg_cost

        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.cost, var_list=self.params)

        self.pred_class = tf.arg_min(self.y_pred,dimension=0)
        self.true_class = tf.placeholder(tf.int64,shape=[1], name="class")

        self.corr_class = tf.equal(self.pred_class,self.true_class)

    def train_on_batch(self,session,x1,x2,y_true,lr=0.01, batch_size=128):
        res = []
        for ind in range(0,len(x1),batch_size):
            res.append(session.run([self.optimizer,self.class_cost,self.enc1,self.enc2,self.reg_cost],feed_dict={
                self.x1:x1[ind:ind+batch_size],
                self.x2:x2[ind:ind+batch_size],
                self.y_true:y_true[ind:ind+batch_size],
                self.lr:lr}))
        return [i[1] for i in res]

    def train_on_sample(self,session,x1,x2,y_true):
        #self.learning_rate*=self.decay
        return session.run([self.optimizer,self.class_cost,self.enc1,self.enc2,self.reg_cost], feed_dict={self.x1:x1,self.x2:x2,self.y_true:y_true, self.lr:self.learning_rate})

    def calc_dist_mat(self,session,x_in, batchsize=128):
        x1 = []
        for i in range(0, len(x_in), batchsize):
            x1.extend(session.run(self.enc1, feed_dict={self.x1: x_in[i:i + batchsize]}))
        arr=[]
        for symb in x1:
            x2 =[symb for i in range(batchsize)]
            tArr = []
            for i in range(0,len(x1),batchsize):
                tArr.extend(np.reshape(session.run(self.y_pred,feed_dict={self.enc1:x1[i:i+batchsize], self.enc2:x2[0:len(x1[i:i+batchsize])]}),(-1)))
            arr.append(tArr)
        return arr

    def test_on_sample(self,session,x1,x2,y_true):
        c=0
        s=0
        numVis = min(len(x1),10)
        pic=np.zeros((self.shape[1]*2,self.shape[2]*numVis))
        for ind in range(len(x1)):
            s+=1

            pred =session.run(self.pred_class,feed_dict={self.x1:x1[ind],self.x2:x2[ind]})
            c+=(pred[0]==y_true[ind])
            if ind<numVis:
                #pic[0:32, 32 * ind:32 * (ind + 1)] = x1[ind][y_true[ind]][:,:,0]
                #pic[32:64, 32 * ind:32 * (ind + 1)] = x2[ind][pred[0]][:,:,0]
                pic[0:self.shape[1]             , self.shape[2] * ind:self.shape[2] * (ind + 1)] = x1[ind][y_true[ind]][:,:,0]
                pic[self.shape[1]:2*self.shape[1], self.shape[2] * ind:self.shape[2] * (ind + 1)] = x2[ind][pred[0]][:,:,0]



        return float(c)/s, pic






from lib import loader

#from lib import vis


#Achieves reasonable performance after a while
#Class rate of about 50% ?
def backup1Net():
    batch = 128
    px = 32
    net = SiameseNetMetric([None, px, px, 1], fcl_layer_size=[], encoding_size=400,
                            conv_layer_size=[20, 40, 60, 100], conv_dim=[7, 5, 3, 3])
    p_same=0.125
    return batch, px, net, p_same


def backup2Net():
    batch = 128
    px = 48
    net = SiameseNetMetric([None, px, px, 1], fcl_layer_size=[], encoding_size=600,
                            conv_layer_size=[20, 40, 60, 100], conv_dim=[7, 5, 3, 3], regularization=True)

    p_same = 0.05
    return batch, px, net, p_same


"""
iteration 320000
seconds since last: 90.0502369404
Cost: 0.0686717076823
RegCost: 0.0379163712896
Class Rate Test: 0.45703125
Class Rate Train: 0.41015625
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseMetricBackup3_320000.ckpt
"""
def backup3Net():
    batch = 256
    px = 64
    net = SiameseNetMetric([None, px, px, 1], fcl_layer_size=[], encoding_size=600,
                            conv_layer_size=[20, 40, 60, 100], conv_dim=[7, 5, 3, 3], regularization=True)

    p_same = 0.05
    return batch, px, net, p_same

def backup4Net():
    batch = 128
    px = 32
    net = SiameseNetMetric([None, px, px, 1], fcl_layer_size=[], encoding_size=480,
                            conv_layer_size=[20, 40, 60, 100], conv_dim=[7, 5, 3, 3], regularization=True)

    p_same = 0.05
    return batch, px, net, p_same


def runInit(backupFunc=backup1Net):
    batch,px,net,_ = backupFunc()
    net.build()
    saver = tf.train.Saver()
    return net,saver

def runRestore(sess,saver,path):
    saver.restore(sess, path)

if __name__ == "__main__":
    batch,px,net,p_same = backup2Net()
    #imVis = vis.ImageVisualiser("Predicted Classes",(px*2,px*10))
    net.build()
    omni = loader.OmniGlotLoader(px)
    init=tf.initialize_all_variables()
    saver = tf.train.Saver()
    iterations = 1000000
    do_init=True

    import time
    with tf.Session() as sess:
        if do_init:
            sess.run(init)
        else:
            saver.restore(sess,"savedNets/SiameseBackup1_%s.ckpt"%(99999))
        cSum=0
        regSum=0
        start = time.time()
        for i in tqdm(range(iterations),miniters=100):
            x1,x2,y=omni.get_training_sample(batch,p_same=p_same)
            res,c,enc1,enc2,reg=net.train_on_sample(sess,x1,x2,y)
            cSum+=c
            regSum+=reg
            if i%1000==0:
                print "iteration %s"%i
                print "seconds since last: %s"%(time.time()-start)
                start=time.time()
                x1_test,x2_test,y_test = omni.get_testing_sample(batch)
                print "Cost: %s"%(cSum/1000)
                print "RegCost: %s"%(regSum/1000)
                regSum=0
                cSum=0
                e,im = net.test_on_sample(sess,x1_test,x2_test,y_test)
                #imVis.add_data(im)
                #imVis.show()
                print "Class Rate Test: %s"%e
                x1_test, x2_test, y_test = omni.get_testing_sample(batch,testing_set="False")
                e2, _ = net.test_on_sample(sess, x1_test, x2_test, y_test)
                print "Class Rate Train: %s" %e2

            if i%10000==0 or i==iterations-1:
                #pass
                save_path = saver.save(sess, folder_path+"savedNets/SiameseMetricBackup2_%s.ckpt"%i)
                print("Model saved in file: %s" % save_path)