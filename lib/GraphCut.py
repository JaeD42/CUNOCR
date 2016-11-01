import networkx as nx
import numpy as np
from scipy import ndimage

class GraphCut(object):

    def __init__(self,sigma=0.8):
        self.sigma=sigma


    def calc_cut(self,img,source_rect=None,sink_rect=None):
        G=nx.DiGraph()
        width,height=img.shape

        if source_rect==None:
            source_rect=[0,width/3,0,height]

        if sink_rect==None:
            sink_rect=[2*width/3,width,0,height]

        img_gaus = ndimage.filters.gaussian_filter(img, self.sigma, mode='nearest')


        for x in range(width):
            for y in range(height):

                if x>source_rect[0] and x<source_rect[1] and y>source_rect[2] and y<source_rect[3]:
                    G.add_edge("source",(x,y))

                if x > sink_rect[0] and x < sink_rect[1] and y > sink_rect[2] and y < sink_rect[3]:
                    G.add_edge((x, y), "sink")

                for i in range(-1,2):
                    for j in range(-1,2):
                        if (i!=0 or j!=0) and x+i>0 and x+i<width and y+j>0 and y+j<height and img_gaus[x+i,y+j]!=1:
                            G.add_edge((x,y),(x+i,y+j),{'weight':min(1-img_gaus[x,y],1-img_gaus[x+i,y+j])})


        cut_value,partition = nx.minimum_cut(G,'source','sink',capacity='weight')

        xs = np.array([p[0] for p in partition[0]])
        ys = np.array([p[1] for p in partition[0]])

        return partition

        """
        xs2 = xs-min(xs)
        ys2 = ys-min(ys)

        img_l = np.ones((max(xs2),max(ys2)))

        for ind in range(len(xs)):
            img_l[xs2[ind],ys2[ind]]=img[xs[ind],ys[ind]]


        xs = np.array([p[0] for p in partition[1]])
        ys = np.array([p[1] for p in partition[1]])

        xs2 = xs - min(xs)
        ys2 = ys - min(ys)

        img_r = np.ones((max(xs2),max(ys2)))

        for ind in range(len(xs)):
            img_r[xs2[ind],ys2[ind]]=img[xs[ind],ys[ind]]





        return img_l,img_r
        """

    def calc_up_down(self,img):
        width, height = img.shape
        return self.calc_cut(img,source_rect=[0,width,0,height/3],sink_rect=[0,width,2*height/3, height])

    def calc_left_right(self,img):
        width, height = img.shape
        return self.calc_cut(img, source_rect=[0, width/3, 0, height], sink_rect=[2*width/3, width, 0, height])


