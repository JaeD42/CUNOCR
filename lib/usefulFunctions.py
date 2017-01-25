
import os
import sys

"""
make sure we are on Path
"""

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"
import numpy as np


def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def getErrorMat(pred, truth, lim=0.5):
    """
    Create Error matrix from prediction and truth values
    :param pred:
    :param truth:
    :param lim:
    :return:
    """
    pred = np.array(pred)
    truth = np.array(truth)

    TP = np.sum((pred < lim) * (truth == 0))
    TN = np.sum((pred > lim) * (truth != 0))
    FN = np.sum((pred > lim) * (truth == 0))
    FP = np.sum((pred < lim) * (truth != 0))

    return {"TP":TP, "TN":TN, "FP":FP, "FN":FN}

def dists_encs(x,arr,net,session,batchsize=128):
    dists=[]
    rep_x = np.repeat([x],batchsize,axis=0)
    for i in range(0,len(arr),128):
        i_e = min(i+128,len(arr))
        dists.extend(session.run(net.y_pred,feed_dict={net.enc1:arr[i:i_e],net.enc2:rep_x[:i_e-i]}))
    return dists


def getEncoding(imgList,net,session):
    encs = []
    for i in range(0,len(imgList),128):
        encs.extend(session.run(net.enc1,feed_dict={net.x1:imgList[i:i+128]}))
    return encs

#def getDistsFromEncs(enc1,encList,net,session)


def getFolderPath():
    return folder_path


if __name__=="__main__":
    print getFolderPath()