from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x,**kwargs):
        return x
from collections import deque

class CharacterExtraction(object):


    def __init__(self,image):
        pass

def search(page,classes,x_start,y_start,classNum,saved_dict, size=1):
    a=range(-size,size+1)
    q = deque()
    q.append([x_start,y_start])
    width,height = page.shape
    while(q):

        x,y = q.pop()

        for i in a:
            for j in a:
                if i+x>=width or j+y>=height or i+x<0 or j+y<0:
                    pass
                else:
                    if (i!=0 or j!=0) and page[x+i,y+j] and classes[x+i,y+j]==0:
                        classes[x+i,y+j]=classNum
                        saved_dict[classNum].append([x+i,y+j])
                        q.append([x+i,y+j])

    return classes,saved_dict

def run(copy_region=False,px=32, do_dist=True, page_Path = "/home/jan/Desktop/Cuneiform/page clean.png"):
    prePage = np.array(misc.imread(page_Path,"L"))
    page=np.array(prePage,dtype="float32")/255



    width,height = page.shape
    classes=np.zeros((width,height))

    classNum=1

    lim=0.2
    binPage = (1-page)>lim

    sav = {}
    for x in range(width):
        for y in range(height):
            if binPage[x,y] and classes[x,y]==0:
                classes[x,y]=classNum
                sav[classNum]=[[x,y]]
                classes,sav = search(binPage,classes,x,y,classNum,sav,size=2)
                classNum+=1


    print classNum

    all_img=[]
    position=[]
    for i in range(1,classNum):
        vals = sav[i]
        if len(vals)>10:
            xs = np.array([a[0] for a in vals])
            ys = np.array([a[1] for a in vals])
            min_xs = min(xs)
            min_ys = min(ys)
            xs = xs-min(xs)
            ys = ys-min(ys)
            si = (max(xs)+1,max(ys)+1)
            if si[0]>20 and si[1]>20:
                ima = np.ones(si)
                if copy_region:
                    for x in range(si[0]):
                        for y in range(si[1]):
                            ima[x,y]=page[min_xs+x,min_ys+y]
                else:
                    for j in range(len(xs)):
                        ima[xs[j],ys[j]]=page[min_xs+xs[j],min_ys+ys[j]]

                all_img.append(ima)
                position.append((min_xs,min_ys))


    imgs = np.array([np.reshape(np.array(misc.imresize(i, (px, px)), dtype="float32"), (px, px, 1)) for i in all_img])


    if not do_dist:
        return imgs,all_img,position

    if do_dist:

        #import Nets.SiameseNet as sNet

        import Nets.Networks as sNet
        import tensorflow as tf

        import time

        import Evaluation.Evals as evalFunc




        #path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_2_fin2.ckpt"
        path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_Cun_50.ckpt"
        path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackupMetric4_Cun_100.ckpt"
        path = "/home/jan/Desktop/Cuneiform/fold3/SiameseNetClassic_funcbackup4NetTest_48.ckpt"
        path = "/home/jan/Desktop/Cuneiform/fold5/SiameseNetMetric_px64_enc600_Fin.ckpt"
        import matplotlib.pyplot as plt
        #net,saver = sNet.runInit(sNet.backup3Net)
        net,saver = sNet.runInit(sNet.backup3Net)
        print "starting calculations"
        s=time.time()
        with tf.Session() as sess:
            sNet.runRestore(sess, saver, path)
            print "restored"
            scores = evalFunc.pairwise_dist(imgs,net,sess)
        print "calcs took %s seconds"%(time.time()-s)
        return imgs,scores,all_img,position


import os

def extract_images_to_folder(folder_path,page_paths = ["/home/jan/Desktop/Cuneiform/page clean.png"]):
    import Nets.Networks as sNet
    import tensorflow as tf

    import time

    import Evaluation.Evals as evalFunc

    path = "/home/jan/Desktop/Cuneiform/fold3/SiameseNetClassic_funcbackup4NetTest_48.ckpt"
    #path = "/home/jan/Desktop/Cuneiform/savedNets/SiameseMetricBackup2_240000.ckpt"

    import matplotlib.pyplot as plt
    #net, saver = sNet.runInit(sNet.backup2Net)
    net,saver = sNet.runInit(sNet.backup4Net)
    print "starting calculations"
    #s = time.time()

    sess = tf.Session()
    sNet.runRestore(sess, saver, path)


    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    imgs = []

    print "extracting"
    for ind,p in tqdm(enumerate(page_paths)):
        img_small,img_big,pos = run(copy_region=False,do_dist=False,page_Path=p,px=48)
        imgs.extend(img_small)




    print "calc scores"
    scores = evalFunc.pairwise_dist(imgs,net,sess)

    print "cluster"
    cScipy = evalFunc.Clustering_Scipy(scores, method="complete")

    labels = cScipy.get_clusters(0.50)
    count = [list(labels).count(i) for i in range(max(labels) + 1)]
    print "save"
    for i in tqdm(range(max(labels)+1)):
        if count[i]>2 and not os.path.exists(folder_path+"%s/"%i):
            os.makedirs(folder_path+"%s/"%i)

    indice = [0 for i in range(max(labels)+1)]
    for ind,val in tqdm(enumerate(labels),total=len(labels)):
        if count[val]>2:
            misc.imsave(folder_path+"%s/%s.png"%(val,indice[val]),imgs[ind][:,:,0])
            indice[val]+=1


if  __name__ == "__main__":

    #imNums = ["0%s"%i for i in range(4,10)]
    imNums = ["%s"%i for i in range(24,54)]
    #imNums.extend(imNums2)
    extract_images_to_folder("/home/jan/Desktop/Cuneiform/img/All_Img_All/", ["/home/jan/Desktop/Cuneiform/pages/page-%s.png"%i for i in imNums])


