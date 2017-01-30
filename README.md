# CUNOCR
Optical Character Recognition of Cuneiform Autographs

Datasets are contained in folder Data/Datasets as zip file


##Requirements
- python 2.7
- tensorflow 0.10 (should also work on newer versions)
- scipy
- sklearn if you want mds plots
- omniglot dataset if training from zero
- tqdm if you want progression bars


##Use

Create a Network by supplying the NetCreator class with needed parameters and load it with a session
```python
from Nets.NetCreator import NetCreator as NetCr
from Nets.Networks import SiameseNetClassic as SiamClass
import tensorflow as tf

netCreator = NetCr(batch_size=128,px=48,netClass=SiamClass)
sess=tf.Session()
#If a saved net is available
#netCreator.loadNet(sess,pathToSavedNet)
netCreator.initialize()
```

To test and train use the NetTrainer class together with datasets
(change path to datasets if needed)
```python
from Nets.NetTrainer import NetTrainer as Trainer
from lib.DataSetLoader import OmniGlotLoader, CuneiformSetLoader
import lib.usefulFunctions as u_func

trainingSet = CuneiformSetLoader(48, u_func.getFolderPath() + "TrainingSet")
testingSet = CuneiformSetLoader(48, u_func.getFolderPath() + "TestingSet")
trainer = Trainer(netCreator.net,sess,trainingSet,testingSet)
for i in range(25):
    print "Iteration %s finished" % (i)
    x = netTrainer.test_epoch(128, epoch=i, save=True)
    print x["TP"], x["TN"], x["FP"], x["FN"]
    x = netTrainer.train_epoch(128)

    netCreator.saveNet(sess, folder_name="foldClassic/", addendum="Test_%s" % i)

```
To visualize created distances use

```python
from Evaluation.ImageCreator import ImageCreator

imgCreator = ImageCreator(netCreator.net,sess)
distanceMatrix,classes = imgCreator.CalcDistMat(testingSet.dataset,[4,5,6,7,8])
imgCreator.MDSPlot(distanceMatrix,cols=classes)
```


### Note
Most files have a __main__ implementation which will probably fail
as the code has massively changed over time. Much experimentation was
done in the notebooks but is absolutely not commented (abandon all hope ye...)

NetCreator & Trainer work fine, but be careful when training complete epochs 
as this will take a long time. 


