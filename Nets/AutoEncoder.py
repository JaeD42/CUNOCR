import tensorflow as tf
import numpy as np
import ops
import matplotlib.pyplot as plt
from NetClass import Net
from tqdm import  tqdm

"""
AutoEncoder class, oriented along the github document of
https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/autoencoder.py
"""

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
training_epochs = 5
batch_size = 256
display_step = 1
examples_to_show = 10

class AutoEncoder(Net):

    def encoder(self, inp):

        w_shape1 = [3,3, self.in_shape[3], 10]
        W1,b1=ops.conv_weight_bias(w_shape1)
        layer1 = ops.max_pool_2x2(tf.nn.relu(ops.conv_2d(inp,W1,b1)))

        w_shape2 = [3,3, 10, 20]
        W2, b2 = ops.conv_weight_bias(w_shape2)
        layer2 = ops.max_pool_2x2(tf.nn.relu(ops.conv_2d(layer1, W2, b2)))

        w_shape3 = [3,3, 20, 40]
        W3, b3 = ops.conv_weight_bias(w_shape3)
        layer3 = ops.max_pool_2x2(tf.nn.relu(ops.conv_2d(layer2, W3, b3)))

        w_shape4 = [3,3, 40, 50]
        W4, b4 = ops.conv_weight_bias(w_shape4)
        layer4 = ops.max_pool_2x2(tf.nn.relu(ops.conv_2d(layer3, W4, b4))) #2x2 when starting with 32x32

        W_fc1, b_fc1 = ops.conv_weight_bias([2*2*50, self.enc_size])


        flat_layer = tf.reshape(layer4, [-1, 2*2*50])



        self.out_enc = tf.nn.sigmoid(tf.matmul(flat_layer, W_fc1) + b_fc1)
        self.encoder_params = [W1, b1, W2, b2, W3, b3, W4, b4, W_fc1, b_fc1]

        return self.out_enc


    def decoder(self, inp):
        current = inp

        W_fc1, b_fc1 = ops.conv_weight_bias([self.enc_size, 2 * 2 * 50])


        pre_rel = tf.matmul(current, W_fc1) + b_fc1

        pre_res = tf.nn.relu(pre_rel)

        pre_up = tf.reshape(pre_res,[-1,2,2,50])

        pre_conv = tf.image.resize_images(pre_up,4,4)
        W1,b1 = ops.conv_weight_bias([3,3,50,40])
        layer1 = tf.nn.relu(ops.conv_2d(pre_conv,W1,b1))

        pre_conv2 = tf.image.resize_images(layer1, 8, 8)
        W2, b2 = ops.conv_weight_bias([3, 3, 40, 20])
        layer2 = tf.nn.relu(ops.conv_2d(pre_conv2, W2, b2))

        pre_conv3 = tf.image.resize_images(layer2, 16, 16)
        W3, b3 = ops.conv_weight_bias([3, 3, 20, 10])
        layer3 = tf.nn.relu(ops.conv_2d(pre_conv3, W3, b3))

        pre_conv4 = tf.image.resize_images(layer3, 32, 32)
        W4, b4 = ops.conv_weight_bias([3, 3, 10, 1])
        self.dec_out = tf.nn.sigmoid(ops.conv_2d(pre_conv4, W4, b4))

        self.decoder_params = [W_fc1,b_fc1,W1,b1,W2,b2,W3,b3,W4,b4]

        return self.dec_out


    def build(self,**inputArgs):
        self.encoder_op=self.encoder(self.x)
        self.decoder_op = self.decoder(self.encoder_op)
        self.y_pred = self.decoder_op
        self.y_true = self.x
        self.cost = tf.reduce_mean(tf.pow(self.y_pred - self.y_true, 2))
        self.params = self.encoder_params
        self.params.extend(self.decoder_params)
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost, var_list=self.params)




    def __init__(self,img_shape, enc_size=128):
        self.filters=[10]
        self.in_shape = img_shape
        self.x_in = tf.placeholder(tf.float32, shape=[None,784])
        self.x = tf.image.resize_images(tf.reshape(self.x_in,[-1,28,28,1]),32,32)
        self.enc_size=enc_size
        self.enc = tf.placeholder(tf.float32, shape=[img_shape[0],self.enc_size])
        self.learning_rate= 0.01


    def train_on_sample(self,session,feed_dict):
        return session.run([self.optimizer, self.cost], feed_dict=feed_dict)

    def apply(self,session,images):
        return session.run(
                self.y_pred, feed_dict={self.x_in:images})

    def encode(self,session,images):
        return session.run(
            self.encoder_op, feed_dict={self.x_in:images})

    def decode(self,session,encoding):
        return session.run(
            self.decoder_op, feed_dict={self.encoder_op:encoding})








a= AutoEncoder([None,32,32,1])
a.build()

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    # Training cycle
    for epoch in tqdm(range(training_epochs)):
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = a.train_on_sample(sess,{a.x_in: batch_xs})
        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1),
                  "cost=", "{:.9f}".format(c))

    print("Optimization Finished!")

    # Applying encode and decode over test set
    encode_decode = a.apply(sess,mnist.test.images[:examples_to_show])

    # Compare original images with their reconstructions
    f, a = plt.subplots(2, 10, figsize=(10, 2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
        a[1][i].imshow(np.reshape(encode_decode[i], (32, 32)))
    f.show()
    plt.draw()
    plt.waitforbuttonpress()