import time
sTime = time.time()

import lib.SplitPage as splitP
reload(splitP)
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
px=48
import skimage
import skimage.color
import scipy.cluster as sciCl
from Evaluation import Evals as evals
page_number = "04"


golden = (1 + 5 ** 0.5) / 2
def color_img(page,positions,all_img,labels):
    hsv_page = skimage.color.rgb2hsv(np.dstack((page,page,page)))
    for ind,img in enumerate(all_img):
        col = (golden*(labels[ind]+1))%1
        width,height = img.shape
        x_s,y_s = positions[ind]
        for x in range(width):
            for y in range(height):
                if img[x,y]!=1:
                    hsv_page[x_s+x,y_s+y,:]=[col,1,0.5]
    return skimage.color.hsv2rgb(hsv_page)
imgs,scores,all_img,u_l_positions = splitP.run(px=px, page_Path="/home/jan/Desktop/Cuneiform/pages/page-%s.png"%page_number)
page = misc.imread("/home/jan/Desktop/Cuneiform/pages/page-%s.png"%page_number, mode="L")

cScipy = evals.Clustering_Scipy(scores,method="average")
labels = cScipy.get_clusters(0.8)
print max(labels)
#print labels
count = [list(labels).count(i) for i in range(max(labels)+1)]
print count
labels2 = [i if count[i]>1 else max(labels)+1 for i in labels ]
labels2 = [i-min(labels2) for i in labels2]
count2 = [list(labels2).count(i) for i in range(max(labels2)+1)]
print count2
print len(count2)
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D
mds = manifold.MDS(n_components=2, metric=True)
nPos = mds.fit_transform(scores)
plt.scatter(nPos[:,0],nPos[:,1],c=labels2)
plt.savefig("/home/jan/Desktop/Cuneiform/img/MDS_page_%s.png"%page_number)
img_c = color_img(page,u_l_positions,all_img,labels2)
misc.imsave("/home/jan/Desktop/Cuneiform/img/page-%s_Color.png"%page_number,img_c)

def showNum(ind,labels,imgs,plot=True,px=32):
    tShow = [imgs[i] for i in range(len(labels)) if labels[i]==ind]
    im = np.zeros((px,px*len(tShow)))
    for i in range(len(tShow)):
        im[:,px*i:px*(i+1)]= tShow[i][...,0]
    if plot:
        plt.imshow(im,cmap="Greys")
        plt.show()
    else:
        return im

import os
if not os.path.exists("/home/jan/Desktop/Cuneiform/img/Clusters/page_%s"%page_number):
    os.makedirs("/home/jan/Desktop/Cuneiform/img/Clusters/page_%s"%page_number)
for i in range(0,max(labels2)):
    misc.imsave("/home/jan/Desktop/Cuneiform/img/Clusters/page_%s/cluster%s.png"%(page_number,i),showNum(i,labels2,imgs,False,px=px))

print "used time: %s seconds"%(time.time()-sTime)