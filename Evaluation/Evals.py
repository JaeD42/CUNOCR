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

    def __init__(self,dist_mat,method="single"):
        self.dist_mat = dist_mat
        self.dist_vec = []
        for i in range(len(dist_mat)-1):
            self.dist_vec.extend(dist_mat[i,i+1:])
        self.clusters = sciCl.hierarchy.linkage(self.dist_vec,method=method)

    def get_linkages(self):
        return self.clusters

    def get_clusters(self,max_cost):
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
    if not os.path.exists(path[:path.rfind("/")]):
        os.makedirs(path[:path.rfind("/")])
    with open(path,'a') as file:
        file.write(str(txt)+"\n")


def pairwise_dist(imgs,net,session):
    #print "pdist"
    #imgs = [np.reshape(np.array(misc.imresize(i,(32,32)),dtype="float32"),(32,32,1)) for i in imgs]
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

"""
Additional Training 1:
0 7674.0 0.0499049 0.0899196218225
1 954.0 0.0312828 0.0112356592904
2 840.0 0.0239131 0.0101812357268
3 802.0 0.0195065 0.00966107774367
4 776.0 0.0165675 0.00927283859927
5 754.0 0.0144829 0.00898430853184
6 718.0 0.0129134 0.00873754985673
7 710.0 0.0116878 0.00854420181346
8 710.0 0.0106846 0.0083866616812
9 702.0 0.00983549 0.00825340531789
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_10.ckpt
10 692.0 0.00911511 0.00813907386465
11 686.0 0.00849354 0.0080386830768
12 680.0 0.00796764 0.00794811966033
13 666.0 0.00751316 0.0078615955189
14 654.0 0.00711504 0.00777996881853
15 648.0 0.00676407 0.00770070064644
16 640.0 0.00645065 0.00762900261708
17 636.0 0.00617029 0.00756396520732
18 634.0 0.00592021 0.00750119540487
19 634.0 0.0056944 0.00744064993173
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_20.ckpt
20 630.0 0.00549095 0.00738506462112
21 626.0 0.00530371 0.00732853037251
22 620.0 0.00512898 0.00727698470715
23 612.0 0.00496752 0.0072292716848
24 614.0 0.00482117 0.00718345842721
25 600.0 0.00468754 0.00713588709326
26 600.0 0.00456463 0.00709506562969
27 596.0 0.00444994 0.00705379890794
28 596.0 0.00434291 0.00701584126895
29 596.0 0.00424327 0.00697940416319
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_30.ckpt
30 590.0 0.00414879 0.00694590646492
31 592.0 0.00405977 0.00691238843571
32 592.0 0.00397671 0.00688138249181
33 590.0 0.00389778 0.00685205801346
34 586.0 0.00382348 0.00682331680897
35 586.0 0.00375287 0.00679612064829
36 584.0 0.00368596 0.00676734309917
37 576.0 0.0036219 0.00674057526118
38 578.0 0.00356097 0.00671272954288
39 578.0 0.00350264 0.00668736713375
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_40.ckpt
40 576.0 0.0034468 0.00666247047438
41 576.0 0.00339335 0.00663978456746
42 574.0 0.00334105 0.0066160999071
43 574.0 0.00329013 0.00659547928571
44 570.0 0.00323961 0.00657492150696
45 568.0 0.00319049 0.0065571804376
46 562.0 0.00314348 0.00653980555832
47 558.0 0.00309926 0.00652238349541
48 560.0 0.00305667 0.00650380155461
49 556.0 0.0030156 0.00648655512941
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_50.ckpt
50 552.0 0.00297596 0.0064694037215
51 550.0 0.00293783 0.0064516718555
52 544.0 0.00290065 0.00643526693751
53 544.0 0.00286483 0.00641823489542
54 536.0 0.00282987 0.0064017271337
55 534.0 0.00279591 0.00638590741995
56 536.0 0.00276254 0.00637046896874
57 538.0 0.00272974 0.00635336999361
58 538.0 0.00269792 0.00633871261989
59 536.0 0.00266698 0.00632443441813
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_60.ckpt
60 538.0 0.00263694 0.00631042188272
61 536.0 0.00260734 0.00629724809878
62 536.0 0.00257845 0.00628470322587
63 536.0 0.00255032 0.00627217710639
64 534.0 0.00252294 0.00626080164673
65 534.0 0.00249624 0.00624875107394
66 532.0 0.00247037 0.00623794602397
67 532.0 0.00244504 0.00622748156175
68 532.0 0.00242027 0.0062163336576
69 530.0 0.00239572 0.00620502030791
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_70.ckpt
70 530.0 0.00237169 0.00619415294432
71 528.0 0.00234792 0.00618230926076
72 528.0 0.00232461 0.00617150427848
73 526.0 0.00230173 0.00615999596971
74 522.0 0.00227923 0.00614824048111
75 520.0 0.00225734 0.00613754436615
76 518.0 0.00223614 0.00612696013128
77 520.0 0.00221571 0.00611649266753
78 518.0 0.00219586 0.00610565971017
79 514.0 0.00217665 0.00609567781596
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_80.ckpt
80 512.0 0.00215794 0.00608484608657
81 512.0 0.00213983 0.00607592049983
82 512.0 0.00212226 0.0060658719632
83 512.0 0.00210509 0.00605712753324
84 512.0 0.00208861 0.0060476968926
85 510.0 0.00207238 0.00603903183465
86 508.0 0.00205658 0.00602951926477
87 506.0 0.00204119 0.0060215369252
88 504.0 0.00202618 0.00601261989401
89 504.0 0.00201158 0.00600402155212
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_90.ckpt
90 504.0 0.00199723 0.00599602564352
91 498.0 0.00198315 0.00598784847456
92 498.0 0.00196943 0.00597960893174
93 498.0 0.0019561 0.00597137027957
94 498.0 0.00194293 0.00596439351313
95 498.0 0.00192985 0.00595643590387
96 498.0 0.00191728 0.00594820484788
97 498.0 0.0019048 0.00594093942636
98 498.0 0.0018927 0.00593328258238
99 498.0 0.00188073 0.00592622908683
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackup4_ExtraCuneiform_100.ckpt
100 498.0 0.00186895 0.00591814678007



Additional Metric:
0 3390.0 0.0525265 0.0410352463928
False Negatives: 480
False Positives: 2910
True Negatives: 63650
True Positives: 560
Took 99.806843996 seconds
1 828.0 0.0434053 0.0107797266996
False Negatives: 680
False Positives: 148
True Negatives: 66412
True Positives: 360
Took 99.8232591152 seconds
2 816.0 0.0405599 0.0103322787023
False Negatives: 694
False Positives: 122
True Negatives: 66438
True Positives: 346
Took 98.7732508183 seconds
3 802.0 0.0387101 0.0100709590948
False Negatives: 688
False Positives: 114
True Negatives: 66446
True Positives: 352
Took 98.2983319759 seconds
4 802.0 0.037359 0.0100693873144
False Negatives: 682
False Positives: 120
True Negatives: 66440
True Positives: 358
Took 97.8068799973 seconds
5 802.0 0.0364758 0.0100018461316
False Negatives: 682
False Positives: 120
True Negatives: 66440
True Positives: 358
Took 97.3194701672 seconds
6 804.0 0.03565 0.00997417989464
False Negatives: 686
False Positives: 118
True Negatives: 66442
True Positives: 354
Took 97.3453090191 seconds
7 796.0 0.0347791 0.00987052015856
False Negatives: 690
False Positives: 106
True Negatives: 66454
True Positives: 350
Took 97.297380209 seconds
8 794.0 0.0340549 0.00976540560452
False Negatives: 688
False Positives: 106
True Negatives: 66454
True Positives: 352
Took 97.355036974 seconds
9 788.0 0.0334694 0.00968989979126
False Negatives: 686
False Positives: 102
True Negatives: 66458
True Positives: 354
Took 97.3429911137 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_10.ckpt
10 784.0 0.0329516 0.00963175091756
False Negatives: 686
False Positives: 98
True Negatives: 66462
True Positives: 354
Took 97.6136040688 seconds
11 790.0 0.0324858 0.0095943041234
False Negatives: 690
False Positives: 100
True Negatives: 66460
True Positives: 350
Took 97.3059809208 seconds
12 792.0 0.0320901 0.00951224192908
False Negatives: 696
False Positives: 96
True Negatives: 66464
True Positives: 344
Took 97.2694079876 seconds
13 796.0 0.0317299 0.00946488459607
False Negatives: 700
False Positives: 96
True Negatives: 66464
True Positives: 340
Took 97.299503088 seconds
14 790.0 0.0313108 0.00944251704401
False Negatives: 696
False Positives: 94
True Negatives: 66466
True Positives: 344
Took 97.2936761379 seconds
15 796.0 0.0309405 0.0094311506678
False Negatives: 698
False Positives: 98
True Negatives: 66462
True Positives: 342
Took 97.2690489292 seconds
16 796.0 0.0306239 0.00941935774616
False Negatives: 700
False Positives: 96
True Negatives: 66464
True Positives: 340
Took 97.26054883 seconds
17 790.0 0.0303086 0.00939213847838
False Negatives: 692
False Positives: 98
True Negatives: 66462
True Positives: 348
Took 97.3024308681 seconds
18 778.0 0.0300427 0.00935406203802
False Negatives: 688
False Positives: 90
True Negatives: 66470
True Positives: 352
Took 97.3362560272 seconds
19 776.0 0.0297841 0.00931042508932
False Negatives: 686
False Positives: 90
True Negatives: 66470
True Positives: 354
Took 97.2779510021 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_20.ckpt
20 776.0 0.0294895 0.00928605036323
False Negatives: 692
False Positives: 84
True Negatives: 66476
True Positives: 348
Took 97.6136169434 seconds
21 774.0 0.0292513 0.00927401319844
False Negatives: 692
False Positives: 82
True Negatives: 66478
True Positives: 348
Took 97.3299920559 seconds
22 774.0 0.0289581 0.00921993006156
False Negatives: 692
False Positives: 82
True Negatives: 66478
True Positives: 348
Took 97.2766368389 seconds
23 774.0 0.0286648 0.0091693104531
False Negatives: 684
False Positives: 90
True Negatives: 66470
True Positives: 356
Took 97.2432560921 seconds
24 766.0 0.0284146 0.00913805619729
False Negatives: 678
False Positives: 88
True Negatives: 66472
True Positives: 362
Took 97.2826850414 seconds
25 764.0 0.0281538 0.00910346670631
False Negatives: 678
False Positives: 86
True Negatives: 66474
True Positives: 362
Took 97.2378790379 seconds
26 766.0 0.0279307 0.00906262698377
False Negatives: 680
False Positives: 86
True Negatives: 66474
True Positives: 360
Took 97.269698143 seconds
27 768.0 0.0277429 0.0090282078761
False Negatives: 682
False Positives: 86
True Negatives: 66474
True Positives: 358
Took 97.2613499165 seconds
28 764.0 0.0275588 0.0089879395896
False Negatives: 682
False Positives: 82
True Negatives: 66478
True Positives: 358
Took 97.3085310459 seconds
29 764.0 0.0273298 0.00895188883626
False Negatives: 682
False Positives: 82
True Negatives: 66478
True Positives: 358
Took 97.3033351898 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_30.ckpt
30 764.0 0.0271284 0.00893695380421
False Negatives: 680
False Positives: 84
True Negatives: 66476
True Positives: 360
Took 97.6231210232 seconds
31 758.0 0.0269497 0.00892587815592
False Negatives: 678
False Positives: 80
True Negatives: 66480
True Positives: 362
Took 97.2943079472 seconds
32 756.0 0.0267545 0.00889832927054
False Negatives: 678
False Positives: 78
True Negatives: 66482
True Positives: 362
Took 97.2529039383 seconds
33 756.0 0.0265091 0.00886383867321
False Negatives: 680
False Positives: 76
True Negatives: 66484
True Positives: 360
Took 97.2955160141 seconds
34 750.0 0.0262794 0.00882527410316
False Negatives: 678
False Positives: 72
True Negatives: 66488
True Positives: 362
Took 97.2967419624 seconds
35 750.0 0.0260876 0.00880768279234
False Negatives: 674
False Positives: 76
True Negatives: 66484
True Positives: 366
Took 97.2514050007 seconds
36 748.0 0.0259379 0.00880386790392
False Negatives: 672
False Positives: 76
True Negatives: 66484
True Positives: 368
Took 97.2513709068 seconds
37 750.0 0.0257891 0.00878031912911
False Negatives: 676
False Positives: 74
True Negatives: 66486
True Positives: 364
Took 97.2398478985 seconds
38 752.0 0.0256606 0.00876403996178
False Negatives: 676
False Positives: 76
True Negatives: 66484
True Positives: 364
Took 97.2859380245 seconds
39 752.0 0.0255341 0.00873189514512
False Negatives: 676
False Positives: 76
True Negatives: 66484
True Positives: 364
Took 97.2799091339 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_40.ckpt
40 742.0 0.0253883 0.00868533291234
False Negatives: 670
False Positives: 72
True Negatives: 66488
True Positives: 370
Took 97.6038649082 seconds
41 740.0 0.0252326 0.00865377979819
False Negatives: 670
False Positives: 70
True Negatives: 66490
True Positives: 370
Took 97.3017311096 seconds
42 742.0 0.0250867 0.00863472255725
False Negatives: 672
False Positives: 70
True Negatives: 66490
True Positives: 368
Took 97.2537930012 seconds
43 734.0 0.0249577 0.00862298393802
False Negatives: 664
False Positives: 70
True Negatives: 66490
True Positives: 376
Took 97.3150119781 seconds
44 734.0 0.0248306 0.00861439636907
False Negatives: 662
False Positives: 72
True Negatives: 66488
True Positives: 378
Took 97.3179969788 seconds
45 730.0 0.0246953 0.00861731214605
False Negatives: 660
False Positives: 70
True Negatives: 66490
True Positives: 380
Took 97.2449700832 seconds
46 726.0 0.0245468 0.0086097184464
False Negatives: 656
False Positives: 70
True Negatives: 66490
True Positives: 384
Took 97.308701992 seconds
47 730.0 0.0244155 0.00859853377058
False Negatives: 660
False Positives: 70
True Negatives: 66490
True Positives: 380
Took 97.2521500587 seconds
48 724.0 0.0242793 0.0085808450567
False Negatives: 658
False Positives: 66
True Negatives: 66494
True Positives: 382
Took 97.2941520214 seconds
49 722.0 0.02413 0.00855106735614
False Negatives: 660
False Positives: 62
True Negatives: 66498
True Positives: 380
Took 97.3135399818 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_50.ckpt
50 720.0 0.0240272 0.00853227803996
False Negatives: 658
False Positives: 62
True Negatives: 66498
True Positives: 382
Took 97.64772892 seconds
51 718.0 0.0239351 0.00851849924267
False Negatives: 656
False Positives: 62
True Negatives: 66498
True Positives: 384
Took 97.2778949738 seconds
52 720.0 0.0238294 0.00851921697121
False Negatives: 660
False Positives: 60
True Negatives: 66500
True Positives: 380
Took 97.2534930706 seconds
53 716.0 0.0237305 0.00851605207095
False Negatives: 656
False Positives: 60
True Negatives: 66500
True Positives: 384
Took 97.3019859791 seconds
54 720.0 0.0236341 0.0085144525458
False Negatives: 656
False Positives: 64
True Negatives: 66496
True Positives: 384
Took 97.2850310802 seconds
55 720.0 0.0235422 0.00849599791298
False Negatives: 654
False Positives: 66
True Negatives: 66494
True Positives: 386
Took 97.2425169945 seconds
56 716.0 0.023459 0.00847163744279
False Negatives: 652
False Positives: 64
True Negatives: 66496
True Positives: 388
Took 97.297796011 seconds
57 720.0 0.0233761 0.00845942659185
False Negatives: 654
False Positives: 66
True Negatives: 66494
True Positives: 386
Took 97.2317659855 seconds
58 718.0 0.0232803 0.00844404478017
False Negatives: 652
False Positives: 66
True Negatives: 66494
True Positives: 388
Took 97.3725690842 seconds
59 722.0 0.0231625 0.00842591153077
False Negatives: 654
False Positives: 68
True Negatives: 66492
True Positives: 386
Took 97.3288550377 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_60.ckpt
60 718.0 0.023062 0.00841761764776
False Negatives: 650
False Positives: 68
True Negatives: 66492
True Positives: 390
Took 97.6329779625 seconds
61 720.0 0.0229803 0.00840933912959
False Negatives: 652
False Positives: 68
True Negatives: 66492
True Positives: 388
Took 97.282886982 seconds
62 722.0 0.0229033 0.00839551561904
False Negatives: 654
False Positives: 68
True Negatives: 66492
True Positives: 386
Took 97.2918179035 seconds
63 724.0 0.0228213 0.00837596163718
False Negatives: 656
False Positives: 68
True Negatives: 66492
True Positives: 384
Took 97.3660709858 seconds
64 722.0 0.022743 0.00836263002104
False Negatives: 656
False Positives: 66
True Negatives: 66494
True Positives: 384
Took 97.2928578854 seconds
65 720.0 0.0226766 0.00835642034422
False Negatives: 654
False Positives: 66
True Negatives: 66494
True Positives: 386
Took 97.2714819908 seconds
66 724.0 0.0226044 0.00833834486577
False Negatives: 652
False Positives: 72
True Negatives: 66488
True Positives: 388
Took 97.2828090191 seconds
67 722.0 0.0225225 0.00832243398996
False Negatives: 656
False Positives: 66
True Negatives: 66494
True Positives: 384
Took 97.2405278683 seconds
68 720.0 0.0224213 0.00830883716019
False Negatives: 656
False Positives: 64
True Negatives: 66496
True Positives: 384
Took 97.2886891365 seconds
69 720.0 0.022334 0.0082943473579
False Negatives: 652
False Positives: 68
True Negatives: 66492
True Positives: 388
Took 97.2861168385 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_70.ckpt
70 718.0 0.0222491 0.00826501831103
False Negatives: 652
False Positives: 66
True Negatives: 66494
True Positives: 388
Took 97.6092998981 seconds
71 722.0 0.0221647 0.00825513050557
False Negatives: 656
False Positives: 66
True Negatives: 66494
True Positives: 384
Took 97.3110029697 seconds
72 722.0 0.0220804 0.00824120656741
False Negatives: 656
False Positives: 66
True Negatives: 66494
True Positives: 384
Took 97.2363607883 seconds
73 714.0 0.0219778 0.00822204304843
False Negatives: 652
False Positives: 62
True Negatives: 66498
True Positives: 388
Took 97.282171011 seconds
74 712.0 0.0218894 0.00820462673266
False Negatives: 652
False Positives: 60
True Negatives: 66500
True Positives: 388
Took 97.2936680317 seconds
75 706.0 0.0218237 0.00818528125712
False Negatives: 648
False Positives: 58
True Negatives: 66502
True Positives: 392
Took 97.2632620335 seconds
76 702.0 0.0217573 0.00817665096794
False Negatives: 646
False Positives: 56
True Negatives: 66504
True Positives: 394
Took 97.2725448608 seconds
77 704.0 0.0216764 0.00816894911004
False Negatives: 646
False Positives: 58
True Negatives: 66502
True Positives: 394
Took 97.243532896 seconds
78 698.0 0.0215969 0.00816759614583
False Negatives: 644
False Positives: 54
True Negatives: 66506
True Positives: 396
Took 97.3168160915 seconds
79 702.0 0.021525 0.00816493898117
False Negatives: 650
False Positives: 52
True Negatives: 66508
True Positives: 390
Took 97.2868859768 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_80.ckpt
80 700.0 0.0214656 0.00814444081634
False Negatives: 648
False Positives: 52
True Negatives: 66508
True Positives: 392
Took 97.6096260548 seconds
81 698.0 0.0214075 0.00812001480826
False Negatives: 642
False Positives: 56
True Negatives: 66504
True Positives: 398
Took 97.2843039036 seconds
82 700.0 0.0213675 0.0081063616779
False Negatives: 642
False Positives: 58
True Negatives: 66502
True Positives: 398
Took 97.2489020824 seconds
83 702.0 0.0213294 0.00809470114136
False Negatives: 646
False Positives: 56
True Negatives: 66504
True Positives: 394
Took 97.3145501614 seconds
84 704.0 0.0212899 0.0080848893938
False Negatives: 646
False Positives: 58
True Negatives: 66502
True Positives: 394
Took 97.2948470116 seconds
85 700.0 0.021254 0.00807763433577
False Negatives: 646
False Positives: 54
True Negatives: 66506
True Positives: 394
Took 97.2877571583 seconds
86 702.0 0.0212226 0.00806929539929
False Negatives: 648
False Positives: 54
True Negatives: 66506
True Positives: 392
Took 97.3335258961 seconds
87 696.0 0.0211891 0.00806164377664
False Negatives: 642
False Positives: 54
True Negatives: 66506
True Positives: 398
Took 97.2890148163 seconds
88 688.0 0.0211565 0.00805475963344
False Negatives: 634
False Positives: 54
True Negatives: 66506
True Positives: 406
Took 97.3042128086 seconds
89 686.0 0.0211198 0.00804889129419
False Negatives: 636
False Positives: 50
True Negatives: 66510
True Positives: 404
Took 97.298853159 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_90.ckpt
90 688.0 0.0210831 0.00804577328977
False Negatives: 638
False Positives: 50
True Negatives: 66510
True Positives: 402
Took 97.6305720806 seconds
91 688.0 0.0210453 0.00803633268354
False Negatives: 638
False Positives: 50
True Negatives: 66510
True Positives: 402
Took 97.2912709713 seconds
92 686.0 0.0210075 0.00802680667502
False Negatives: 638
False Positives: 48
True Negatives: 66512
True Positives: 402
Took 97.2786159515 seconds
93 688.0 0.0209663 0.00801910036517
False Negatives: 642
False Positives: 46
True Negatives: 66514
True Positives: 398
Took 97.2828891277 seconds
94 688.0 0.0209052 0.0080154084018
False Negatives: 638
False Positives: 50
True Negatives: 66510
True Positives: 402
Took 97.2693610191 seconds
95 684.0 0.0208467 0.00801656712584
False Negatives: 636
False Positives: 48
True Negatives: 66512
True Positives: 404
Took 97.2820188999 seconds
96 690.0 0.0207946 0.00800926500481
False Negatives: 642
False Positives: 48
True Negatives: 66512
True Positives: 398
Took 97.3078439236 seconds
97 692.0 0.0207472 0.0080052644542
False Negatives: 642
False Positives: 50
True Negatives: 66510
True Positives: 398
Took 97.2585930824 seconds
98 694.0 0.0206972 0.00800412651993
False Negatives: 644
False Positives: 50
True Negatives: 66510
True Positives: 396
Took 97.3118050098 seconds
99 692.0 0.0206564 0.00799650602925
False Negatives: 644
False Positives: 48
True Negatives: 66512
True Positives: 396
Took 97.3211419582 seconds
Model saved in file: /home/jdisselhoff/Cuneiform/savedNets/SiameseBackupMetric2_ExtraCuneiform_100.ckpt
100 690.0 0.0206199 0.0079881697033
False Negatives: 644
False Positives: 46
True Negatives: 66514
True Positives: 396
Took 97.6771509647 seconds
"""