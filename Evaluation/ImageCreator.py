import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold
from collections import Counter
from Evaluation import Evals as evals
import lib.usefulFunctions as u_func
from tqdm import tqdm
import random
from collections import Counter

class ImageCreator(object):
    """
    Class for easier creation of plots and validation of networks
    """
    def __init__(self,net,session):
        """
        Class for creation of plots using the network
        and validating performance
        :param net: A siamese network
        :param session: TF session
        """
        font = {'family': 'normal',
                'size': 18}

        matplotlib.rc('font', **font)
        self.net=net
        self.session=session
        self.agglomClusters = None

    @staticmethod
    def ROCPlot(paths,names):
        """
        Create ROC plot from saved prediction files
        :param paths:
        :param names:
        :return:
        """
        xs = []
        ys = []
        for filePath in tqdm(paths):
            arr = []
            with open(filePath, "r") as f:
                arr = f.read()
            print arr[0:1000]
            arr = arr.replace("array([ ", "")
            arr = arr.replace("], dtype=float32)", "")

            arr = arr.split("]")
            p = [float(i) for i in arr[0][1:].split(",")]
            t = [int(i == " True") for i in arr[1][1:].split(",")]

            vals = np.array([p, t])

            vals.sort()
            x = [0 for i in range(100)]
            y = x[:]
            x2 = x[:]
            y2 = x[:]

            for i in range(100):
                confMat = u_func.getErrorMat(p, t, lim=float(i) / 99)
                x[i] = float(confMat["FP"]) / (confMat["FP"] + confMat["TN"])
                y[i] = float(confMat["TP"]) / (confMat["TP"] + confMat["FN"])
                if (confMat["TP"] + confMat["FP"]) == 0:
                    x2[i] = 1
                else:
                    x2[i] = float(confMat["TP"]) / (confMat["TP"] + confMat["FP"])
                y2[i] = float(confMat["TP"]) / (confMat["TP"] + confMat["FN"])
            xs.append(x)
            ys.append(y)

        plt.title("ROC Curve")
        for ind in range(len(xs)):
            plt.plot(xs[ind], ys[ind], label=names[ind], linewidth=2.5)
        plt.legend(loc=4)
        plt.ylim(0, 1.1)
        plt.xlim(0, 1.1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.show()

    @staticmethod
    def MDSPlot(distmat,cols,size=20,**kwargs):
        """
        Create MDS plot from distance matrix
        :param distmat:
        :param cols:
        :param size:
        :param kwargs:
        :return:
        """
        mds = manifold.MDS(n_components=2, metric=True)
        nPos = mds.fit_transform(distmat)
        plt.scatter(nPos[:, 0], nPos[:, 1], c=cols, s=size,**kwargs)
        plt.show()

    @staticmethod
    def ClusterPurityPlot(clusterLabels,networkType,clusteringType,classes,classsize_on_x=False):
        """
        Cluster purity plot with labels
        :param clusterLabels:
        :param networkType:
        :param clusteringType:
        :param classes:
        :param classsize_on_x:
        :return:
        """
        n = zip(classes, clusterLabels)

        numbers = []
        for i in range(max(clusterLabels) + 1):
            temps = [j[0] for j in n if j[1] == i]
            numbers.append(Counter(temps))

        plotVals = {}
        plotCount = {}

        for c in numbers:
            val = c.most_common(1)

            s = sum(c.values())
            if classsize_on_x:
                s2= sum(np.array(classes)==val[0][0])
            else:
                s2=s

            if s2 in plotVals:
                plotCount[s2] = plotCount[s2] + 1
                plotVals[s2] = plotVals[s2] + float(val[0][1]) / s
            else:
                plotCount[s2] = 1
                plotVals[s2] = float(val[0][1]) / s

        xys = []
        for val in plotVals.keys():
            xys.append([val, float(plotVals[val]) / plotCount[val]])
        xys.sort(key=lambda x: x[0])
        xys = np.array(xys)
        x = xys[:, 0]
        y = xys[:, 1]
        plt.plot(x, y, "-", x, y, "or")
        if classsize_on_x:
            plt.xlabel("Class size")
        else:
            plt.xlabel("Cluster size")
        plt.ylabel("Average purity")
        plt.title(str(clusteringType)+" clustering, "+str(networkType)+" network")
        plt.ylim((0, 1.1))
        plt.show()

    def CalcDistMat(self,data,indices):
        """
        Helper function for distmat calculation
        :param data: list of list of characters (inner lists are classes)
        :param indices: indices of classes to use
        :return:
        """
        dataList = []
        classes = []
        for ind, val in enumerate(indices):
            dataList.extend(data[val])
            classes.extend([ind] * len(data[val]))
        dMat = np.array(self.net.calc_dist_mat(session=self.session, x_in=dataList))
        return dMat, classes

    def PreCalcClusterLabels(self,dMat,method="complete"):
        self.agglomClusters = evals.Clustering_Scipy(dMat,method=method)

    def CutNumCLusterLavels(self,numClusters):
        start = 0
        stop = 1
        mid=0.5
        clusters = self.agglomClusters.get_clusters(mid)
        while max(clusters)!=numClusters:
            if max(clusters)<numClusters:
                stop=mid
                mid=(start+stop)/2
            else:
                start=mid
                mid = (start + stop) / 2
            clusters = self.agglomClusters.get_clusters(mid)
        return clusters


    def CutoffClusterLabels(self,cutoff):
        return self.agglomClusters.get_clusters(cutoff)

    def CalcClusterLabels(self,dMat,cutoff,method="complete"):
        cScipy = evals.Clustering_Scipy(dMat, method=method)
        return cScipy.get_clusters(cutoff)

    def FirstNNPerformance(self,data,numCharacters,numExperiments,weighted = False,retAllPreds=False):
        dataLength = len(data)
        correct=0
        allPreds=[]
        for i in range(numExperiments):
            if weighted:
                s=set()
                w = [len(i) for i in data]
                w=[float(i)/sum(w) for i in w]
                while len(s)<numCharacters:
                    s.update(np.random.choice(range(dataLength),size=numCharacters,p=w))
                sampled = list(s)
                np.random.shuffle(sampled)
                sampled=sampled[0:numCharacters]

            else:
                sampled = random.sample(range(dataLength),numCharacters)
            trueClass = sampled[0]

            tmp = random.sample(range(len(data[trueClass])),2)

            trueChar = [data[trueClass][tmp[0]] for i in range(numCharacters)]
            testSet = [data[trueClass][tmp[1]]]

            for i in sampled[1:]:
                testSet.append(random.sample(data[i],1)[0])

            pred = self.session.run(self.net.y_pred,feed_dict={self.net.x1:trueChar,self.net.x2:testSet})
            if retAllPreds:
                allPreds.append(pred)
            correct+=(np.argmin(pred)==0)
        if retAllPreds:
            return float(correct)/numExperiments,allPreds
        return float(correct)/numExperiments


    def KNNPerformance(self,dataLong,numCharacters,numExperiments,k=3):
        data = [i for i in dataLong if len(i)>k]
        dataLength = len(data)
        correct = 0
        for i in range(numExperiments):
            sampled = random.sample(range(dataLength), numCharacters)
            trueClass = sampled[0]

            tmp = random.sample(range(len(data[trueClass])), 1+k)

            trueChar = [data[trueClass][tmp[0]] for i in range(k*numCharacters)]
            testSet = [data[trueClass][i] for i in tmp[1:]]

            for i in sampled[1:]:
                testSet.extend(random.sample(data[i], k))

            pred = self.session.run(self.net.y_pred, feed_dict={self.net.x1: trueChar, self.net.x2: testSet})

            classes = [i for j in range(k) for i in range(numCharacters)]
            zipped = zip(pred,classes)
            zipped.sort(key=lambda x:x[0])
            #print zipped
            c=Counter([zipped[i][1] for i in range(k)])

            correct += (c.most_common(1)[0][0] == 0)

        return float(correct) / numExperiments


    def KNNPerformancePlot(self,data):
        pass


if __name__=="__main__":
    size = 100
    ps = np.random.random((size,5))
    ps[0:size/2,:]+=0.8
    dMat = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            dMat[i,j]=np.mean((ps[i,:]-ps[j,:])**2)

    ImageCreator.MDSPlot(dMat,[0]*(size/2)+[1]*(size/2))