import networkx as nx
import numpy as np
from scipy import ndimage

class GraphCut(object):
    """
    Mincut algorithm for images
    Should work, but was not used in thesis
    """

    def __init__(self,sigma=0.8):
        """

        :param sigma: Sigma for gaussian filter to make image more connected
        """
        self.sigma=sigma


    def calc_cut(self,img,source_rect=None,sink_rect=None):
        """
        Calcualte Graph cut of img based on a source and sink rectangle
        :param img: Image to split
        :param source_rect: Pixels in this rect count as source (x,y,x+w,y+w)
        :param sink_rect: Pixels in this rect count as sink (x,y,x+w,y+w)
        :return: List of two lists containing pixel indices of the resulting cut images
        """
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



    def calc_left_right(self,img):
        width, height = img.shape
        return self.calc_cut(img,source_rect=[0,width,0,height/3],sink_rect=[0,width,2*height/3, height])

    def calc_up_down(self,img):
        width, height = img.shape
        return self.calc_cut(img, source_rect=[0, width/3, 0, height], sink_rect=[2*width/3, width, 0, height])


if __name__ == "__main__":
    GC = GraphCut(sigma=0.9)
    from scipy import misc
    img = misc.imread("GraphCutEx1.png",mode='L')/255.0
    print img
    shape= img.shape
    part = GC.calc_up_down(img)
    print part[0]
    print part[1]
    imgCol = np.zeros((shape[0],shape[1],3))
    for z in part[0]:
        if type(z)!=tuple:
            continue
        x,y=z[:]

        imgCol[x,y,0]=1-img[x,y]
    for z in part[1]:
        if type(z)!=tuple:
            continue
        x,y=z[:]
        imgCol[x,y,1]=1-img[x,y]

    import matplotlib.pyplot as plt
    plt.imshow(imgCol)
    plt.show()