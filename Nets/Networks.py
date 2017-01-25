import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"
import tensorflow as tf
from NetClass import Net
import ops


class SiameseNetClassic(Net):

    def build(self,**inputArgs):
        """
        Builds the network, x1 and x2 are the inputs, enc1 and enc2 the respective encodings
        y_pred is our prediction while y_true is the real value if given
        cost gives the combined costs and optimizer our learning method

        :param inputArgs:
        :return:
        """

        self.lr = tf.placeholder(tf.float32, shape=[])
        self.x1 = tf.placeholder(tf.float32, shape=self.shape, name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=self.shape, name="x2")

        self.y_true = tf.placeholder(tf.float32, shape=[self.shape[0]])

        with tf.variable_scope("image_filters") as scope:
            """
            Create bot encodings
            """
            self.enc1=self.build_encoding(self.x1)
            scope.reuse_variables()
            self.enc2=self.build_encoding(self.x2)

        self.params = self.enc_weights
        self.dec_weights = ops.conv_weight_bias([self.encoding_size,1])

        self.params.extend(self.dec_weights)
        self.y_pred = tf.reshape(tf.nn.sigmoid(tf.matmul(tf.abs(self.enc1-self.enc2), self.dec_weights[0]) + self.dec_weights[1]),[-1])

        self.class_cost = tf.reduce_mean(
            -tf.mul(self.y_true, tf.log(tf.clip_by_value(self.y_pred, 0.0001, 0.9999)))) - tf.reduce_mean(
            tf.mul(1-self.y_true, tf.log(1 - tf.clip_by_value(self.y_pred, 0.0001, 0.9999))))

        self.cost = self.class_cost
        self.reg_cost = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*self.reg_constant


        if self.regularization:
            self.cost+=self.reg_cost

        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.cost, var_list=self.params)

        self.pred_class = tf.arg_min(self.y_pred,dimension=0)
        self.true_class = tf.placeholder(tf.int64,shape=[1], name="class")

        self.corr_class = tf.equal(self.pred_class,self.true_class)



class SiameseNetMetric(Net):



    def build(self,**inputArgs):
        """
                Builds the network, x1 and x2 are the inputs, enc1 and enc2 the respective encodings
                y_pred is our prediction while y_true is the real value if given
                cost gives the combined costs and optimizer our learning method

                :param inputArgs:
                :return:
        """

        self.lr = tf.placeholder(tf.float32, shape=[])
        self.x1 = tf.placeholder(tf.float32, shape=self.shape, name="x1")
        self.x2 = tf.placeholder(tf.float32, shape=self.shape, name="x2")

        self.y_true = tf.placeholder(tf.float32, shape=[self.shape[0]])

        with tf.variable_scope("image_filters") as scope:
            self.enc1=self.build_encoding(self.x1)
            scope.reuse_variables()
            self.enc2=self.build_encoding(self.x2)

        self.params = self.enc_weights
        self.dec_weights = ops.conv_weight_bias([self.encoding_size,1])

        self.params.extend(self.dec_weights)

        """
        ONLY DIFFERENCE BETWEEN METRIC AND CLASSICAL
        last weights are squared to ensure positivity, and we use tf.nn.tanh insteand of tf.nn.sigmoid
        """
        self.y_pred = tf.reshape(tf.nn.tanh(tf.matmul(tf.abs(self.enc1-self.enc2), tf.square(self.dec_weights[0]))),[-1])

        self.class_cost = tf.reduce_mean(
            -tf.mul(self.y_true, tf.log(tf.clip_by_value(self.y_pred, 0.0001, 0.9999)))) - tf.reduce_mean(
            tf.mul(1-self.y_true, tf.log(1 - tf.clip_by_value(self.y_pred, 0.0001, 0.9999))))

        self.cost = self.class_cost
        self.reg_cost = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))*self.reg_constant


        if self.regularization:
            self.cost+=self.reg_cost

        self.optimizer = tf.train.AdagradOptimizer(self.lr).minimize(self.cost, var_list=self.params)

        self.pred_class = tf.arg_min(self.y_pred,dimension=0)
        self.true_class = tf.placeholder(tf.int64,shape=[1], name="class")

        self.corr_class = tf.equal(self.pred_class,self.true_class)





