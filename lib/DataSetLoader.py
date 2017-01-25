import numpy as np
import time
import os
from scipy import misc

class LanguageLoader(object):

    def __init__(self,dataset,px=48):
        """

        :param dataset: img dataset in the following format:
            dataset[x][y][z] is a single img in form of (px,px,1) with values between 0 and 1
            x is the selected language
            y is the selected character
            z is the selected instance of the character
        """
        self.dataset = dataset
        self.px=px

    def get_epoch_size(self,language=-1):
        n = 0
        if language==-1:
            for lan in self.dataset:
                temp=0
                for symbs in lan:
                    temp += len(symbs)
                n+=temp*(temp-1)/2
        else:
            for ind,symbs in enumerate(self.dataset[language]):
                n+=len(symbs)
            n=(n*(n-1))/2

        return n

    def iterate_epoch_all_languages(self,batch_size):
        """
        Iterate through all languages and create training samples
        :param batch_size: chunk size to return, last retun can be smaller
        :return:
        """
        example_list=[]
        for lan in self.dataset:
            example_list_lan = []
            for ind, symbs in enumerate(lan):
                example_list_lan.extend(zip([ind] * len(symbs), symbs))

            import itertools
            import random

            example_list.extend(itertools.combinations(example_list_lan, 2))

        random.shuffle(example_list)

        for i in range(0, len(example_list), batch_size):
            exs = example_list[i:i + batch_size]
            x1 = [val[0][1] for val in exs]
            x2 = [val[1][1] for val in exs]
            y = [val[0][0] != val[1][0] for val in exs]
            yield x1, x2, y





    def iterate_epoch(self,batch_size,language=0):
        """
        Yields values of a single dataset languages in an epoch;
        A batch of batchsize is returned in x1,x2,y where
        x1 and x2 are images and y is 0 if they belong to
        the same class
        :param batch_size: size of chunks returned, last chunk can be smaller
        :param language: language to use, -1 uses all languages
        :return:
        """
        if language == -1:
            for val in self.iterate_epoch_all_languages(batch_size):
                yield val
        else:

            lan = self.dataset[language]
            example_list = []
            for ind,symbs in enumerate(lan):
                example_list.extend(zip([ind]*len(symbs),symbs))


            import itertools
            import random

            example_list = itertools.combinations(example_list,2)

            example_list = [val for val in example_list]

            random.shuffle(example_list)

            for i in range(0,len(example_list),batch_size):
                exs = example_list[i:i+batch_size]
                x1 = [val[0][1] for val in exs]
                x2 = [val[1][1] for val in exs]
                y = [val[0][0]!=val[1][0] for val in exs]
                yield x1,x2,y




    def get_symbol(self,example_num,symb_num,language_num, **kwargs):
        """

        :param example_num: Example number of character
        :param symb_num: Symbol Number
        :param language_num: Language Number
        :return: Symbol at dataset[x,y,z] with protection against out of bounds
        """
        used_set =self.dataset

        lanLen = len(used_set)

        language = used_set[min(lanLen-1,language_num)]

        symbLen = len(language)

        symb = language[min(symbLen-1,symb_num)]

        exLen = len(symb)

        example = symb[min(exLen-1,example_num)]
        return example

    def get_many_symbols(self,example_nums, symb_nums, language_nums,**kwargs):
        """
        Convenience wrapper to return a list of symbols at once
        :param example_nums: array containing example numbers
        :param symb_nums: array containing symbol numbers
        :param language_nums: array containing language numbers
        :return: List of symbols
        """
        return [self.get_symbol(example_nums[i], symb_nums[i],language_nums[i], **kwargs)
                for i in range(min(len(example_nums),min(len(language_nums),len(symb_nums))))]

    def get_example_from_every_symb(self,example_num,language_num,**kwargs):
        """
        Returns a single example from every symbol in the language
        :param example_num:
        :param language_num:
        :return:
        """

        used_len = len(self.dataset[language_num])

        return [self.get_symbol(example_num,i,language_num, **kwargs) for i in range(used_len)]

    def get_all_examples_for_symb(self,symb_num,language_num,**kwargs):
        """
        Returns every instance of a specified symbol in a language
        :param symb_num:
        :param language_num:
        :return:
        """
        used_len = len(self.dataset[language_num][symb_num])

        return [self.get_symbol(i, symb_num, language_num, **kwargs) for i in range(used_len)]

    def get_training_sample(self, batchsize, p_same = 0.5, **kwargs):
        """
        Unnecessary wrapper around get_training_sample_from_set
        :param batchsize:
        :param p_same:
        :return:
        """

        used_set = self.dataset

        return self.get_train_sample_from_set(batchsize,used_set,p_same)


    def get_train_sample_from_set(self, batchsize, used_set, p_same=0.5):
        x1 = []
        x2 = []
        y = []

        for i in range(int(p_same * batchsize)):
            rand_lan = used_set[np.random.randint(0, len(used_set))]
            rand_symb = rand_lan[np.random.randint(0, len(rand_lan))]
            randx1, randx2 = np.random.randint(0, len(rand_symb), 2)
            x1.append(rand_symb[randx1])
            x2.append(rand_symb[randx2])
            y.append(0)

        for i in range(batchsize - int(p_same * batchsize)):
            rand_lan1 = used_set[np.random.randint(0, len(used_set))]
            rand_symb1Num = np.random.randint(0, len(rand_lan1))
            rand_symb2Num = np.random.randint(0, len(rand_lan1))
            rand_symb1 = rand_lan1[rand_symb1Num]
            randx1 = np.random.randint(0, len(rand_symb1))
            # rand_lan2 = self.training_samples[np.random.randint(0, len(self.training_samples))]
            rand_symb2 = rand_lan1[rand_symb2Num]
            randx2 = np.random.randint(0, len(rand_symb2))
            x1.append(rand_symb1[randx1])
            x2.append(rand_symb2[randx2])
            y.append(1-int(rand_symb1Num == rand_symb2Num))

        return x1, x2, y


class CuneiformSetLoader(LanguageLoader):
    """
    Loads Cuneiform dataset
    """

    def __init__(self, px, path):
        """

        :param px: size of the images
        :param path: path of the dataset
        """
        self.px = px
        dataset = self.preload(path)

        super(CuneiformSetLoader, self).__init__(dataset)

    def numpyify(self, img):
        return np.reshape(np.array(img, dtype='float32'), (self.px, self.px, 1)) / 255.0

    def preload(self, path):
        language = []
        folders = next(os.walk(path))[1]
        folders.sort()
        folders.sort(key=lambda x: len(x))

        for fold in folders:
            temp = []
            symbs = next(os.walk(os.path.join(path, fold)))[2]
            for symb in symbs:
                temp.append(self.numpyify(
                    misc.imresize(misc.imread(os.path.join(path, fold, symb), mode='L'), (self.px, self.px))))
            language.append(temp)

        dataset = [language]
        return dataset


class OmniGlotLoader(object):
    """
    Class to load Omniglot characters, seems bloated as it contains both
    testing and training data, have to think about how to fix this
    TODO: Make this more concise and short
    """

    def __init__(self,px, path = "Data/Datasets/omniglot-master/images/"):
        self.px = px
        #self.path = path
        self.path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/" + path
        self.training_samples = []
        self.testing_samples = []
        self.pre_load()
        self.training_loader = LanguageLoader(self.training_samples)
        self.testing_loader = LanguageLoader(self.testing_samples)

    def numpyify(self,img):
        return np.reshape(np.array(img,dtype='float32'),(self.px,self.px,1))/255.0

    def pre_load(self):
        t=time.time()
        print "Loading Omniglot Data"
        self.training_samples = []

        for language in os.listdir(self.path+"images_background/"):
            lan = []
            for symbol in os.listdir(self.path+"images_background/"+language):
                symb = []
                for pic in os.listdir(self.path+"images_background/"+language+"/"+symbol):
                    symb.append(self.numpyify(misc.imresize(misc.imread(self.path+"images_background/"+language+"/"+symbol+"/"+pic),(self.px,self.px))))
                lan.append(symb)
            self.training_samples.append(lan)

        self.testing_samples = []
        for language in os.listdir(self.path+"images_evaluation/"):
            lan = []
            for symbol in os.listdir(self.path+"images_evaluation/"+language):
                symb = []
                for pic in os.listdir(self.path + "images_evaluation/" + language + "/" + symbol):
                    symb.append(self.numpyify(misc.imresize(
                        misc.imread(self.path + "images_evaluation/" + language + "/" + symbol + "/" + pic),
                        (self.px, self.px))))
                lan.append(symb)
            self.testing_samples.append(lan)

        print "Loading took %s seconds"%(time.time()-t)



    def get_symbol(self,example_num,symb_num,language_num, use_testing_set=False):
        if use_testing_set:
            return self.testing_loader.get_symbol(example_num,symb_num,language_num)
        else:
            return self.training_loader.get_symbol(example_num,symb_num,language_num)

    def get_many_symbols(self,example_nums, symb_nums, language_nums, use_testing_set=False):
        return [self.get_symbol(example_nums[i], symb_nums[i],language_nums[i],use_testing_set=use_testing_set)
                for i in range(min(len(example_nums),min(len(language_nums),len(symb_nums))))]

    def get_example_from_every_symb(self,example_num,language_num,use_testing_set=False):

        if use_testing_set:
            used_len = len(self.testing_samples[language_num])
        else:
            used_len = len(self.training_samples[language_num])

        return [self.get_symbol(example_num,i,language_num,use_testing_set) for i in range(used_len)]

    def get_all_examples_for_symb(self,symb_num,language_num,use_testing_set=False):
        if use_testing_set:
            used_len = len(self.testing_samples[language_num][symb_num])
        else:
            used_len = len(self.training_samples[language_num][symb_num])

        return [self.get_symbol(i, symb_num, language_num, use_testing_set) for i in range(used_len)]

    def get_training_sample(self, batchsize, p_same = 0.5, testing_set=False):
        x1=[]
        x2=[]
        y =[]


        if testing_set:
            return self.testing_loader.get_training_sample(batchsize,p_same)
        else:
            return self.training_loader.get_training_sample(batchsize,p_same)



    def get_training_sample_with_addition(self,batchsize,add_language_num, p_same=0.5):
        """
        Returns values from Training set combined with values from some testing sets
        to test how easy it is to generalize with little data
        :param batchsize:
        :param add_language_num:
        :param p_same:
        :return:
        """
        used_set = self.training_samples[:]
        addition1 = self.get_example_from_every_symb(0,add_language_num,use_testing_set=True)
        addition2 = self.get_example_from_every_symb(1,add_language_num,use_testing_set=True)

        used_set.append([[addition1[i],addition2[i]] for i in range(len(addition1))])

        return self.get_train_sample_from_set(batchsize,used_set,p_same)


    def get_train_sample_from_set(self, batchsize, used_set, p_same=0.5):
        x1 = []
        x2 = []
        y = []

        for i in range(int(p_same * batchsize)):
            rand_lan = used_set[np.random.randint(0, len(used_set))]
            rand_symb = rand_lan[np.random.randint(0, len(rand_lan))]
            randx1, randx2 = np.random.randint(0, len(rand_symb), 2)
            x1.append(rand_symb[randx1])
            x2.append(rand_symb[randx2])
            y.append(0)

        for i in range(batchsize - int(p_same * batchsize)):
            rand_lan1 = used_set[np.random.randint(0, len(used_set))]
            rand_symb1Num = np.random.randint(0, len(rand_lan1))
            rand_symb2Num = np.random.randint(0, len(rand_lan1))
            rand_symb1 = rand_lan1[rand_symb1Num]
            randx1 = np.random.randint(0, len(rand_symb1))
            # rand_lan2 = self.training_samples[np.random.randint(0, len(self.training_samples))]
            rand_symb2 = rand_lan1[rand_symb2Num]
            randx2 = np.random.randint(0, len(rand_symb2))
            x1.append(rand_symb1[randx1])
            x2.append(rand_symb2[randx2])
            y.append(1-int(rand_symb1Num == rand_symb2Num))

        return x1, x2, y


    def get_testing_sample(self, batchsize, testing_set=True):
        x1 = []
        x2 = []
        y = []

        used_set = self.testing_samples
        if not testing_set:
            used_set = self.training_samples

        for i in range(batchsize):
            rand_lan = used_set[np.random.randint(0, len(used_set))]
            symb_num = np.random.randint(0, len(rand_lan))
            rand_symb = rand_lan[symb_num]
            randx1= np.random.randint(1, len(rand_symb))
            x1.append(np.array([rand_symb[randx1] for symb in rand_lan]))
            x2.append(np.array([symb[0] for symb in rand_lan]))

            y.append(symb_num)

        return x1,x2,y




if __name__ == "__main__":
    CL = CuneiformSetLoader(48,'/home/jan/Desktop/Cuneiform/Data/img/newData27343')
    print CL.get_symbol(0,0,0)

    import matplotlib.pyplot as plt
    plt.imshow(CL.get_symbol(0,0,0)[:,:,0])
    plt.show()

    print len(CL.get_example_from_every_symb(0, 0, use_testing_set=True))