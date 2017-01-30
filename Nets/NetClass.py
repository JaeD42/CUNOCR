import tensorflow as tf
import numpy as np
import ops
class Net(object):

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
        self.build()

    def build_encoding(self, x):
        """
        Builds graph to create encoding from input x
        :param x: input image
        :return: flat layer containing the encoding
        """

        def getVars(name, w_shape):
            """
            Helper function to resuse variables in order to create a siamese net.
            :param name: Name of the variable we want
            :param w_shape: Shape of the variable we want
            :return: Variable with given name if exists, otherwise new variable with given shape
            """
            w = tf.get_variable("W" + name, w_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
            b = tf.get_variable("b" + name, [w_shape[-1]], initializer=tf.constant_initializer(0.1),
                                regularizer=tf.contrib.layers.l2_regularizer(0.01))
            return w, b

        prev_layer = x
        img_size = self.shape[2]

        for ind in range(len(self.conv_layer_size)):
            """
            Iterate through conv_layers and apply convolution, rele and max-pool
            """
            if ind == 0:
                w_shape = [self.conv_dim[ind], self.conv_dim[ind], self.shape[3], self.conv_layer_size[ind]]
            else:
                w_shape = [self.conv_dim[ind], self.conv_dim[ind], self.conv_layer_size[ind - 1],
                           self.conv_layer_size[ind]]
            w, b = getVars("enc%s" % ind, w_shape)
            prev_layer = ops.max_pool_2x2(tf.nn.relu(ops.conv_2d(prev_layer, w, b)))

            self.enc_weights.append(w)
            self.enc_weights.append(b)

        # Reshape for fully connected layers
        next_size = self.conv_layer_size[-1] * img_size / (2 ** len(self.conv_layer_size)) * img_size / (
        2 ** len(self.conv_layer_size))
        flat_layer = tf.reshape(prev_layer, [-1, next_size])

        for ind in range(len(self.fcl_layer_size)):
            """
            Iterate through fully connected layers and apply matmul and sigmoid
            """
            if ind == 0:
                w_shape = [next_size, self.fcl_layer_size[0]]
            else:
                w_shape = [self.fcl_layer_size[ind - 1], self.fcl_layer_size[ind]]
            w, b = getVars("enc_fcl%s" % ind, w_shape)
            flat_layer = tf.nn.sigmoid(tf.matmul(flat_layer, w) + b)
            self.enc_weights.append(w)
            self.enc_weights.append(b)

        return flat_layer

    def build(self,**inputArgs):
        pass


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

    def calc_dist_mat_from_encs(self,session,encs,batchsize=128):
        arr=[]
        for symb in encs:
            x2 =[symb for i in range(batchsize)]
            tArr = []
            for i in range(0,len(encs),batchsize):
                tArr.extend(np.reshape(session.run(self.y_pred,feed_dict={self.enc1:encs[i:i+batchsize], self.enc2:x2[0:len(encs[i:i+batchsize])]}),(-1)))
            arr.append(tArr)
        return arr


    def calc_dist_mat(self,session,x_in, batchsize=128):
        x1 = []
        for i in range(0, len(x_in), batchsize):
            x1.extend(session.run(self.enc1, feed_dict={self.x1: x_in[i:i + batchsize]}))

        return self.calc_dist_mat_from_encs(session,x1,batchsize)

    def test_on_sample(self,session,x1,x2,y_true):
        c=0
        s=0
        numVis = min(len(x1),10)
        pic=np.zeros((self.shape[1]*2,self.shape[2]*numVis))
        for ind in range(len(x1)):
            s+=1

            pred =session.run(self.pred_class,feed_dict={self.x1:x1[ind],self.x2:x2[ind]})
            c+=(pred==y_true[ind])
            if ind<numVis:
                #pic[0:32, 32 * ind:32 * (ind + 1)] = x1[ind][y_true[ind]][:,:,0]
                #pic[32:64, 32 * ind:32 * (ind + 1)] = x2[ind][pred[0]][:,:,0]
                pic[0:self.shape[1]             , self.shape[2] * ind:self.shape[2] * (ind + 1)] = x1[ind][y_true[ind]][:,:,0]
                pic[self.shape[1]:2*self.shape[1], self.shape[2] * ind:self.shape[2] * (ind + 1)] = x2[ind][pred][:,:,0]



        return float(c)/s, pic


