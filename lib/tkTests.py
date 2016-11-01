import Tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
# implement the default mpl key bindings
from matplotlib.backend_bases import key_press_handler

from matplotlib.figure import Figure





matplotlib.use('TkAgg')

root = tk.Tk()
root.configure(background='black')
content = tk.Frame(root)
content.grid(column=0, row=0)
x=[False for i in range(3)]

def switchX(pos,canvas):
    def inner(*args):
        x[pos]=not x[pos]
        if x[pos]:
            canvas.get_tk_widget().configure(background='black', highlightcolor='black', highlightbackground='black')
        else:
            canvas.get_tk_widget().configure(background='red', highlightcolor='red', highlightbackground='red')
        canvas.show()
        root.update()
        print x
    return inner


def addMat(arr,master,column,row):
    f = Figure(figsize=(2,2), dpi=100)
    f.patch.set_alpha(0.0)
    a = f.add_subplot(111)

    a.plot(arr)
    canvas = FigureCanvasTkAgg(f, master=master)
    canvas.show()
    # canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
    canvas.get_tk_widget().grid(column=column, row=row)
    canvas.get_tk_widget().bind('<Button-1>',switchX(row-1,canvas))
    canvas.get_tk_widget().configure(background='red', highlightcolor='red', highlightbackground='red')
    #canvas.get_tk_widget().configure(background='black', highlightcolor='black', highlightbackground='black')
    #one = tk.Checkbutton(content, text="", variable=x[row-1], onvalue=True)
    #one.grid(column=column+1, row=row)

def pr(*args):
    print args[0].widget

addMat([i for i in range(10)],content,1,1)
addMat([i**2 for i in range(10)],content,1,2)
#addMat([i**3 for i in range(10)],content,1,3)
#addMat([i**4 for i in range(10)],content,1,4)

Btn = tk.Button(master=content,text='same', command=pr)
Btn.grid(column=0, row=3)

"""
f = Figure(figsize=(5, 4), dpi=100)
a = f.add_subplot(111)
a.plot([i**2 for i in range(10)])
canvas = FigureCanvasTkAgg(f, master=content)
canvas.show()
#canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas.get_tk_widget().grid(column=2,row=2)

f2 = Figure(figsize=(5, 4), dpi=100)
a2 = f2.add_subplot(111)
a2.plot([i for i in range(10)])
canvas2 = FigureCanvasTkAgg(f2, master=content)
canvas2.show()
#canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
canvas2.get_tk_widget().grid(column=1,row=1)
"""
tk.mainloop()