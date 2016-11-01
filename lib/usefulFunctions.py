
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/"
import numpy as np
def createPath(self, path):
    if not os.path.exists(path):
        os.makedirs(path)


def getErrorMat(pred, truth, lim=0.5):
    pred = np.array(pred)
    truth = np.array(truth)
    print pred.shape
    print truth.shape
    print ((pred < lim) * (truth == 0)).shape
    TP = np.sum((pred < lim) * (truth == 0))
    TN = np.sum((pred > lim) * (truth != 0))
    FN = np.sum((pred > lim) * (truth == 0))
    FP = np.sum((pred < lim) * (truth != 0))

    return {"TP":TP, "TN":TN, "FP":FP, "FN":FN}


def getFolderPath():
    return folder_path


if __name__=="__main__":
    print getFolderPath()