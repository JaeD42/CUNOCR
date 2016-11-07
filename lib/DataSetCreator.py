import lib.loader as Load
import Tkinter as tk
import numpy as np
import os
import time
import matplotlib
import matplotlib.pyplot as plt
from scipy import misc
from collections import deque as DQue

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure

class DatasetCreator(object):

    def createPath(self,path):
        if not os.path.exists(path):
            os.makedirs(path)

    def __init__(self,dataPath,savePath,net,session,px, sort_by=lambda x:np.mean(x)):
        self.sort_by = sort_by
        self.max_imgs_from_datset = 50
        self.px=px

        self.loader = Load.CuneiformSetLoader(px,dataPath)
        self.savePath = savePath
        if not self.savePath[-1]=="/":
            self.savePath=self.savePath+"/"
        self.data = self.loader.dataset[0]
        self.net=net
        self.session =session
        self.encoding = []
        self.similarities = []
        self.root = tk.Tk()
        self.root.wm_title("Data set Creation Gui")


        self.dataInd=0

        matplotlib.use('TkAgg')

    def combine(self):
        self.pre_calc_enc()
        self.similarities = self.calc_sims()
        self.similarities.sort(key=lambda x: self.sort_by(x[2]))
        self.similarities.sort(key=lambda x:min(x[0],x[1]))

        same,dSet =self.run_gui_combine_fast()
        foldNumber = 0

        for isSet,s in zip(dSet.is_set,dSet.sets):
            if isSet:
                p = self.savePath + "%s/" % foldNumber
                self.createPath(p)
                iNum=0
                for data_num in s:

                    for img in self.data[data_num]:
                        misc.imsave(p + str(iNum) + ".png", img[:, :, 0])
                        iNum+=1
                foldNumber+=1


    def split(self,start=0):
        res,dataQue = self.run_gui_split()
        for ind in range(len(res)):
            dat = res[ind]
            p=self.savePath+"%s/"%(start+ind)
            self.createPath(p)
            for ind2,val in enumerate(dat):
                misc.imsave(p+str(ind2)+".png",val[:,:,0])

        p2=self.savePath+"rest%s/"%start
        ind=0
        while len(dataQue)!=0:
            ps=p2+str(ind)+"/"
            self.createPath(ps)
            d=dataQue.pop()
            for i,img in enumerate(d):
                misc.imsave(ps+str(i)+".png",img[:,:,0])
            ind+=1




    def reshape_with_zeros(self,arr, shape):
        ret = np.zeros(shape)
        arr = np.array(arr)
        ret[0:arr.shape[0] * self.px, 0:arr.shape[1]] = np.reshape(arr, (-1, len(arr[0])))
        return ret

    def hstack_img_list(self,arr1, arr2):
        mLen = max(len(arr1), len(arr2)) * self.px
        rs = self.reshape_with_zeros
        return np.hstack((rs(arr1, (mLen, len(arr1[0]))), rs(arr2, (mLen, len(arr2[0])))))

    def pre_calc_enc(self):
        print "Calculating encodings of all Images to save time"
        t=time.time()
        n = self.net
        for d in self.data:
            d=[np.reshape(misc.imresize(i[:,:,0],(48,48)),(48,48,1)) for i in d]
            self.encoding.append(self.session.run(n.enc1,feed_dict={n.x1:d[:self.max_imgs_from_datset]}))

        print "Encodings done in %s seconds"%(time.time()-t)

    def get_sim(self,ind1,ind2):
        encs = self.encoding
        net = self.net

        e1 = np.repeat(encs[ind1], len(encs[ind2]), axis=0)
        e2 = np.tile(encs[ind2], (len(encs[ind1]), 1))
        sim = self.session.run(net.y_pred, feed_dict={net.enc1: e1, net.enc2: e2})
        return sim

    def calc_sims(self):
        print "Calculating Similarities of Folders"
        t = time.time()
        s = []
        for i in range(int(len(self.data)/(1))):
            for j in range(i, int(len(self.data)/(1))):
                if i != j:
                    s.append([i, j, self.get_sim(i, j)])

        print "Similarities done in %s seconds"%(time.time()-t)

        return s

    def run_gui_split(self):
        global res,destCont,dataItem
        resDat = []

        res = []

        content = tk.Frame(self.root)
        content.grid(column=0,row=0)
        destCont = tk.Frame(content)
        destCont.grid(column=0, row=0)

        dataQue = DQue(self.data)

        dataItem = dataQue.pop()


        def on_key_event(event):

            if event.char=='o':
                btnPress()
            if event.char=='d':
                allDiff()
            if event.char=='x':
                removeAll()



        def switchRes(pos, canvas):
            def inner(*args):
                res[pos] = not res[pos]
                if res[pos]:
                    canvas.get_tk_widget().configure(background='black', highlightcolor='black',
                                                     highlightbackground='black')
                else:
                    canvas.get_tk_widget().configure(background='red', highlightcolor='red', highlightbackground='red')
                canvas.show()
                self.root.update()
                print res

            return inner

        def addMat(arr, master, column, row):
            f = Figure(figsize=(1, 1), dpi=100)
            f.patch.set_alpha(0.0)
            a = f.add_subplot(111)

            a.imshow(arr[:,:,0])
            canvas = FigureCanvasTkAgg(f, master=master)
            canvas.show()
            # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            canvas.get_tk_widget().grid(column=column, row=row)
            canvas.get_tk_widget().bind('<Button-1>', switchRes(row - 1, canvas))
            canvas.get_tk_widget().configure(background='black', highlightcolor='black', highlightbackground='black')
            # canvas.get_tk_widget().configure(background='black', highlightcolor='black', highlightbackground='black')
            # one = tk.Checkbutton(content, text="", variable=x[row-1], onvalue=True)
            # one.grid(column=column+1, row=row)

        def showData(list,master):
            global res
            for ind,d in enumerate(list):
                addMat(d,master,1,ind+1)
            res=[True for i in range(len(list))]

        def showNext():
            global destCont,dataItem
            destCont.destroy()
            destCont = tk.Frame(content)
            destCont.grid(column=0, row=0)
            if len(dataQue)>0:
                print "Rest len %s"%len(dataQue)
                d = dataQue.pop()

                while len(d)==1 and len(dataQue)>0:
                    resDat.append(d)
                    d=dataQue.pop()
                if len(d)>6:
                    dataQue.append(d[6:])
                    d=d[:6]
                    dataItem=d
                    showData(d,destCont)
                else:
                    dataItem=d
                    showData(d,destCont)

        def btnPress():
            trs = []
            fls = []
            print res
            for i in range(len(res)):
                if res[i]:
                    trs.append(dataItem[i])
                else:
                    fls.append(dataItem[i])
            resDat.append(trs)
            if len(fls)>2:
                dataQue.append(fls)
            else:
                if len(fls)>=1:
                    for val in fls:
                        resDat.append([val])

            showNext()

        def allDiff():
            for val in dataItem:
                resDat.append([val])
            showNext()

        def removeAll():
            showNext()


        self.root.bind("<Key>", func=on_key_event)

        showData(dataItem,destCont)

        button = tk.Button(master=self.root, text='Fin', command=btnPress)
        button.grid(column=0, row=1)

        tk.mainloop()

        return resDat,dataQue

        #WRONG


    def run_gui_combine_fast(self):
        import DisjointSets as dSets
        dSet = dSets.DisjointSets(range(len(self.data)))
        results = [0 for i in range(len(self.similarities))]
        self.dataInd = 0
        self.lastInd = [0]

        f = Figure(figsize=(5, 4), dpi=100)
        a = f.add_subplot(111)
        vals = self.similarities[self.dataInd]
        a.imshow(self.hstack_img_list(self.data[vals[0]][0:10],self.data[vals[1]][0:10]))

        canvas = FigureCanvasTkAgg(f, master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg(canvas, self.root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def show_next():
            while np.mean(self.similarities[self.dataInd][2])>0.3 and len(self.similarities)>self.dataInd+1:
                self.dataInd+=1
            print np.mean(self.similarities[self.dataInd][2])
            self.lastInd.append(self.dataInd)
            self.dataInd=min(len(self.similarities)-1,self.dataInd+1)
            #self.dataInd+=1

            vals = self.similarities[self.dataInd]

            while dSet.in_same(vals[0],vals[1]) and self.dataInd<len(self.similarities)-1:
                self.dataInd = min(len(self.similarities) - 1, self.dataInd + 1)
                vals = self.similarities[self.dataInd]


            a.clear()
            a.imshow(self.hstack_img_list(self.data[vals[0]][0:6], self.data[vals[1]][0:6]))
            canvas.show()
            self.root.update()

        def on_key_event(event):
            print('you pressed %s' % event.key)
            if event.key=='left':

                self.dataInd= self.lastInd[-1]-1
                del self.lastInd[-1]
                show_next()
            if event.key=='right':
                show_next()
            if event.key=='up':
                _same()
            if event.key=='down':
                _differ()

            if event.key=='space':
                _same()
            key_press_handler(event, canvas, toolbar)

        def _same(*args):
            results[self.dataInd]='Yes'
            [x,y,score]=self.similarities[self.dataInd]
            dSet.merge(x,y)
            show_next()

        def _differ(*args):
            results[self.dataInd]='No'
            show_next()

        canvas.mpl_connect('key_press_event', on_key_event)



        def _quit():
            self.root.quit()  # stops mainloop
            self.root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        button = tk.Button(master=self.root, text='same', command=_same)
        button2 = tk.Button(master=self.root, text='different', command=_differ)
        button3 = tk.Button(master=self.root, text='stop', command=_quit)
        button.pack(side=tk.BOTTOM)
        button2.pack(side=tk.BOTTOM)
        button3.pack(side=tk.BOTTOM)

        tk.mainloop()

        return results,dSet


    def run_gui_combine(self):
        import DisjointSets as dSets
        dSet = dSets.DisjointSets(range(len(self.data)))
        results = [0 for i in range(len(self.similarities))]
        self.dataInd = 0

        f = Figure(figsize=(5, 4), dpi=100)
        a = f.add_subplot(111)
        vals = self.similarities[self.dataInd]
        a.imshow(self.hstack_img_list(self.data[vals[0]][0:10],self.data[vals[1]][0:10]))

        canvas = FigureCanvasTkAgg(f, master=self.root)
        canvas.show()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg(canvas, self.root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        def show_next():
            print np.mean(self.similarities[self.dataInd][2])
            self.dataInd=min(len(self.similarities)-1,self.dataInd+1)
            #self.dataInd+=1
            vals = self.similarities[self.dataInd]
            a.clear()
            a.imshow(self.hstack_img_list(self.data[vals[0]][0:6], self.data[vals[1]][0:6]))
            canvas.show()
            self.root.update()

        def on_key_event(event):
            print('you pressed %s' % event.key)
            if event.key=='left':
                self.dataInd= max(self.dataInd-2,-1)
                show_next()
            if event.key=='right':
                show_next()
            if event.key=='up':
                _same()
            if event.key=='down':
                _differ()

            if event.key=='space':
                _same()
            key_press_handler(event, canvas, toolbar)

        def _same(*args):
            results[self.dataInd]='Yes'
            show_next()

        def _differ(*args):
            results[self.dataInd]='No'
            show_next()

        canvas.mpl_connect('key_press_event', on_key_event)



        def _quit():
            self.root.quit()  # stops mainloop
            self.root.destroy()  # this is necessary on Windows to prevent
            # Fatal Python Error: PyEval_RestoreThread: NULL tstate

        button = tk.Button(master=self.root, text='same', command=_same)
        button2 = tk.Button(master=self.root, text='different', command=_differ)
        button3 = tk.Button(master=self.root, text='stop', command=_quit)
        button.pack(side=tk.BOTTOM)
        button2.pack(side=tk.BOTTOM)
        button3.pack(side=tk.BOTTOM)

        tk.mainloop()

        return results


if __name__=="__main__":
    import Nets.SiameseNet as sNet
    import tensorflow as tf
    netPath = "/home/jan/Desktop/Cuneiform/savedNets/SiameseBackup4_Cun_100.ckpt"
    net, saver = sNet.runInit(sNet.backup3Net)
    sess = tf.Session()
    sNet.runRestore(sess, saver, netPath)
    dsc = DatasetCreator("/home/jan/Desktop/Cuneiform/Data/img/newData27346","/home/jan/Desktop/Cuneiform/Data/img/newData27347",net,sess,px=64,sort_by=lambda x:np.mean(x))
    dsc.combine()

