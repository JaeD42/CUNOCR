

class DisjointSets(object):
    """
    Helper class to create Datasets
    """
    def __init__(self,arr=[]):
        """
        Initialize with list
        :param arr:
        """
        self.sets = [set([i]) for i in arr]
        self.is_set = [True for i in arr]

    def in_same(self,a,b):
        """
        Test if two values are in same set
        :param a:
        :param b:
        :return:
        """

        while not self.is_set[b]:
            b=self.sets[b]

        while not self.is_set[a]:
            a=self.sets[a]

        if a==b:
            return True

        else:
            return False

    def merge(self,x,y):
        """
        Merge two sets
        :param x:
        :param y:
        :return:
        """

        while not self.is_set[x]:
            x=self.sets[x]
        while not self.is_set[y]:
            y=self.sets[y]

        self.sets[x].update(self.sets[y])
        self.sets[y]=x
        self.is_set[y]=False







