import os

def createPath(self, path):
    if not os.path.exists(path):
        os.makedirs(path)


def getErrorMat(pred, truth, lim=0.5):
    TP = (sum(sum((pred < lim) * (truth == 0))))
    TN = (sum(sum((pred > lim) * (truth != 0))))
    FN = (sum(sum((pred > lim) * (truth == 0))))
    FP = (sum(sum((pred < lim) * (truth != 0))))

    return {"TP":TP, "TN":TN, "FP":FP, "FN":FN}