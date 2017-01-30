from scipy import misc
import numpy as np
try:
    import skimage
    import skimage.color
except ImportError:
    print "SKImage not found"

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x,**kwargs):
        return x



import scipy.cluster as sciCl
import matplotlib.pyplot as plt
from itertools import product as pairwise_prod
import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/"

class Eval1(object):
    """
    Deprecated, used in very old tests

    """
    def __init__(self,net,session,px):
        self.net=net
        self.session=session
        self.px = px
        self.page=self.loadImg("/home/jan/Desktop/Cuneiform/page.png")
        self.col_page = np.dstack((self.page,self.page,self.page))
        self.l1s = [self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter1.png",True),
                    self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter1_2.png",True),
                    self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter1_3.png",True)]

        self.l2s = [self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter2.png", True),
                    self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter2_2.png", True),
                    self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter2_3.png", True)]

        self.l3s = [self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter3.png", True),
                    self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter3_2.png", True),
                    self.loadImg("/home/jan/Desktop/Cuneiform/CuneiformImg/Letter3_3.png", True)]

    def resize(self,img):
        return np.reshape(misc.imresize(img,(self.px,self.px)),(self.px,self.px,1))

    def loadImg(self,path, resize=False):
        if resize:
            return np.array(self.resize(misc.imread(path,mode="L")),dtype="float32")

        return np.array(misc.imread(path,mode="L"),dtype="float32")/255

    def get_scores(self,x1,x2):
        return self.session.run(self.net.y_pred,feed_dict={self.net.x1:x1,self.net.x2:x2})

    def get_scores_one_v_all(self,xOne,xAll,max_batch=128):
        xMany = np.repeat([xOne],max_batch,axis=0)
        scores=[]
        for i in range(0,len(xAll),128):
            if i+128>len(xAll):
                scores.extend(self.get_scores(xMany[0:len(xAll)-i],xAll[i:]))
            else:
                scores.extend(self.get_scores(xMany,xAll[i:i+128]))
        return scores

    def eval_from_single_char(self,img,char,stepsize=24,imgSize=96,inverted_scores=True):
        imgs=[]
        positions=[]
        for i in range(0,len(img)-max(imgSize,stepsize),stepsize):
            for j in range(0,len(img[0])-max(imgSize,stepsize),stepsize):
                imgs.append(self.resize(img[i:i+imgSize,j:j+imgSize]))
                positions.append((i,j))

        scores = self.get_scores_one_v_all(char,imgs)
        if inverted_scores:
            scores=[1-score for score in scores]

        return scores,positions



    def col_eval_single_char(self,img,char,stepsize=24,imgSize=96,inverted_scores=True):
        scores,positions = self.eval_from_single_char(img,char,stepsize,imgSize,inverted_scores)
        rows, cols = img.shape

        colors = np.zeros((rows, cols, 3))
        for i in range(len(scores)):
            x, y = positions[i]
            colors[x:x + imgSize, y:y + imgSize, 1] += np.ones((imgSize, imgSize)) * scores[i]

        colors = colors / (imgSize / stepsize)
        colors = colors * (colors > 0.3)

        masked = self.mask_img(np.dstack((img, img, img)), colors, alpha=2.1)

        return scores, positions, masked


    def col_eval_three_char(self,img,chars,stepsize=24,imgSize=96,inverted_scores=True):
        scores1,positions = self.eval_from_single_char(img,chars[0],stepsize,imgSize,inverted_scores)
        scores2, positions = self.eval_from_single_char(img, chars[1], stepsize, imgSize, inverted_scores)
        scores3, positions = self.eval_from_single_char(img, chars[2], stepsize, imgSize, inverted_scores)

        rows, cols = img.shape
        colors = np.zeros((rows, cols, 3))
        for i in range(len(scores1)):
            x, y = positions[i]
            colors[x:x + imgSize, y:y + imgSize, 0] += np.ones((imgSize, imgSize)) * scores1[i]
            colors[x:x + imgSize, y:y + imgSize, 0] += np.ones((imgSize, imgSize)) * scores2[i]
            colors[x:x + imgSize, y:y + imgSize, 0] += np.ones((imgSize, imgSize)) * scores3[i]

        colors = colors / (imgSize / stepsize)
        colors = colors/3
        lim = 0.75
        colors = (((colors-lim) * ((colors-lim) > lim)))/(1-lim)

        masked = self.mask_img(np.dstack((img, img, img)), colors, alpha=0.8)

        return [scores1,scores2,scores3], positions, masked



    def mask_img(self,ground_image,mask_img,alpha,num_channels=1):
        temp_ground = skimage.color.rgb2hsv(ground_image)
        temp_mask = skimage.color.rgb2hsv(mask_img)

        temp_ground[...,0] = temp_mask[...,0]
        temp_ground[...,1] = temp_mask[...,1] * alpha * np.sum(mask_img,axis=2)/num_channels

        return skimage.color.hsv2rgb(temp_ground)

    def test_eval_three(self,ls=0):
        lsVals = [self.l1s,self.l2s,self.l3s]

        return self.col_eval_three_char(self.page,lsVals[ls])

    def test_eval(self,ls=1):
        if ls==1:
            return self.col_eval_single_char(self.page,self.l1s[0])
        if ls==2:
            return self.col_eval_single_char(self.page, self.l2s[0])


class Clustering_Scipy(object):
    """
    Class for easier clustering
    """
    def __init__(self,dist_mat,method="single"):
        """
        Class for clustering of data according to distance matrix
        :param dist_mat: distance matrix
        :param method: method for clustering, e.g. single, average, complete or other scipy clusterings
        """
        self.dist_mat = dist_mat
        self.dist_vec = []
        for i in range(len(dist_mat)-1):
            self.dist_vec.extend(dist_mat[i,i+1:])
        self.clusters = sciCl.hierarchy.linkage(self.dist_vec,method=method)

    def get_linkages(self):
        """
        Get result of scipy clustering, unprocessed
        :return:
        """
        return self.clusters

    def get_clusters(self,max_cost):
        """
        Get cluster labels for a certain max cost value after
        which no merging will take place
        :param max_cost: cost at which we should stop merging
        :return: cluster labels
        """
        c_dict={}
        for i in range(len(self.dist_mat)):
            c_dict[i]=[i]

        ind=0
        l = len(self.clusters)
        while ind<l and self.clusters[ind][2]<max_cost:
            new_vals=c_dict.pop(self.clusters[ind][0])
            new_vals.extend(c_dict.pop(self.clusters[ind][1]))
            #print c_dict.keys()
            c_dict[len(self.dist_mat)+ind]=new_vals
            #print c_dict.keys()
            ind+=1
            #print ind

        ind=0
        labels=[-1 for i in range(len(self.dist_mat))]
        for key,value in c_dict.iteritems():
            for val in value:
                labels[val]=ind
            ind+=1
        return labels

def line_to_file(path,txt):
    """
    Deprecated helper function to write a line to a file
    with check if exists and creating if not
    :param path:
    :param txt:
    :return:
    """
    if not os.path.exists(path[:path.rfind("/")]):
        os.makedirs(path[:path.rfind("/")])
    with open(path,'a') as file:
        file.write(str(txt)+"\n")


def pairwise_dist(imgs,net,session):
    """
    Helper function calculating pairwise distances
    Deprecated as network class contains similar function now
    :param imgs:
    :param net:
    :param session:
    :return:
    """
    encs=[]
    for i in tqdm(range(0,len(imgs),128)):
        res = session.run(net.enc1, feed_dict={net.x1: imgs[i:i+128]})
        if len(encs)==0:
            encs=res
        else:
            encs = np.concatenate((encs,res))




    dists=np.zeros((len(imgs),len(imgs)))

    for i in tqdm(range(len(imgs))):
        sc = session.run(net.y_pred, feed_dict={net.enc1:encs, net.enc2:np.repeat([encs[i,:]],len(encs),axis=0)})
        dists[i,:]=np.reshape(sc,(-1))

    return dists

def pairwise_train(imgs,label,net,session, lr,batchsize):
    p_imgs = pairwise_prod(imgs,imgs)
    p_labels = pairwise_prod(label,label)


    costs = []
    z = np.array(zip(p_imgs,p_labels))
    for i in range(0,len(z),128) :
        train_xs = z[i:i+128,0]
        ls=z[i:i+128,1]

        train_x1 = np.array([i for i in train_xs[:,0]])
        train_x2 = np.array([i for i in train_xs[:,1]])

        train_y = [[i] for i in (ls[:,0]==ls[:,1]) ]

        cost = net.train_on_batch(session, train_x1, train_x2, train_y, lr=lr, batch_size=batchsize)
        costs.append(cost)
    #cost = net.train_on_batch(session, train_x1, train_x2, train_y, lr=lr, batch_size=128)
    #costs.append(cost)

    return costs

def test_perf(net,session,loader,lang_num,start_symb,num_tests=5):
    testList = [loader.get_example_from_every_symb(start_symb + i, lang_num, use_testing_set=True) for i in range(num_tests)]
    tests = []
    for test in testList:
        tests.extend(test)

    correct = np.ones((len(testList[0]), len(testList[0])))
    for i in range(len(correct)):
        correct[i, i] = 0
    correct = np.concatenate([correct for i in range(num_tests)])
    correct = np.concatenate([correct for i in range(num_tests)], axis=1)



    dist_mat = np.array(net.calc_dist_mat(session, tests, batchsize=128))

    return dist_mat,correct

def getErrorMat(pred,truth,lim=0.5):
    """
    Generates an error matrix given prediction and truth values
    :param pred:
    :param truth:
    :param lim:
    :return:
    """
    TP = (sum(sum((pred < lim) * (truth == 0)))) - len(truth)
    TN = (sum(sum((pred > lim) * (truth != 0))))
    FN = (sum(sum((pred > lim) * (truth == 0))))
    FP = (sum(sum((pred < lim) * (truth != 0))))

    return TP,TN,FP,FN

def incresed_perf(net,session,loader,lang_num=0,id = "absda"):
    TPFPFile = folder_path+"TrainFiles/TPFP"+id
    ROCFile = folder_path+"TrainFiles/ROC"+id
    line_to_file(TPFPFile,"It,TP,TN,FP,FN,Cost,MSE for 0.5 as boundary")
    RocDists = 0.05
    line_to_file(ROCFile,"It,Equidistant Roc Values %s"%RocDists)

    import time
    num_training_chars = 3

    exs = [loader.get_example_from_every_symb(i,lang_num,use_testing_set=True) for i in range(num_training_chars)]

    train_imgs=[]
    lbls = []
    for ex in exs:
        train_imgs.extend(ex)
        lbls.extend(range(len(ex)))


    train_imgs = np.array(train_imgs)




    num_tests = 2

    testList = [loader.get_example_from_every_symb(num_training_chars+i,lang_num,use_testing_set=True) for i in range(num_tests)]
    tests = []
    for test in testList:
        tests.extend(test)

    correct = np.ones((len(testList[0]),len(testList[0])))
    for i in range(len(correct)):
        correct[i,i]=0

    correct = np.concatenate([correct for i in range(num_tests)])
    correct = np.concatenate([correct for i in range(num_tests)],axis=1)

    lr = 0.001
    for i in range(101):
        start = time.time()
        dist_mat = np.array(net.calc_dist_mat(session,tests, batchsize=128))

        errors = sum(sum(abs((dist_mat>0.5)-correct)))

        if i%10==0 and i>0:
            save_path = saver.save(sess, folder_path + "savedNets/SiameseBackupLap_ExtraCuneiform%s_%s.ckpt" % (id,i))
            print("Model saved in file: %s" % save_path)
        costs = pairwise_train(train_imgs, lbls, net,session,lr, 128)
        lr = lr*0.999

        print i, errors, np.mean(costs), np.mean((dist_mat-correct)**2)

        TP, TN, FP, FN = getErrorMat(dist_mat,correct)

        line_to_file(TPFPFile,"%s,%s,%s,%s,%s,%s,%s"%(i,TP,TN,FP,FN,np.mean(costs),np.mean((dist_mat-correct)**2)))

        RocVals=[]
        for i in np.arange(0,1.00001,RocDists):
            TP, TN, FP, FN = getErrorMat(dist_mat, correct,lim=i)

            TPRate = float(TP)/(TP+FN)
            FPRate = float(FP)/(FP+TN)

            RocVals.append((FPRate,TPRate))

        line_to_file(ROCFile,str(RocVals))

        RocVals = np.array(RocVals)

        plt.plot(RocVals[:,0],RocVals[:,1])
        plt.xlim = (0, 1)
        plt.ylim = (0, 1)
        plt.show()

        print "False Negatives: %s"%(sum(sum((dist_mat > 0.5) * (correct == 0))))
        print "False Positives: %s" % (sum(sum((dist_mat < 0.5) * (correct != 0))))
        print "True Negatives: %s" % (sum(sum((dist_mat > 0.5) * (correct != 0))))
        print "True Positives: %s" % (sum(sum((dist_mat < 0.5) * (correct == 0))))
        print "Took %s seconds"%(time.time()-start)

    return



if __name__ == "__main__":
    #path = folder_path+"savedNets/SiameseMetricBackup3_320000.ckpt"
    #/home/jdisselhoff/Cuneiform/savedNets/SiameseBackup7_200000.ckpt
    path =folder_path+"savedNets/SiameseBackup4_2_fin.ckpt"
    print path
    import Nets.Networks as sNet
    import tensorflow as tf
    import time
    import matplotlib.pyplot as plt
    import lib.DataSetLoader as Loader
    net,saver = sNet.runInit(sNet.backup4Net)
    print "starting calculations"
    s=time.time()
    load = Loader.CuneiformSetLoader(folder_path+"/lib/RotatedDatabase/",48)
    with tf.Session() as sess:
        sNet.runRestore(sess, saver, path)
        incresed_perf(net,sess,load)

