# CUNOCR
Optical Character Recognition of Cuneiform Autographs

Datasets are contained in folder Data/Datasets as zip file


##Requirements

- tensorflow
- scipy
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
