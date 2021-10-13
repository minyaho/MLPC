import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk #NavigationToolbar2TkAg
import numpy as np
from matplotlib.colors import ListedColormap
from mlp import NeuralNetMLP
#import threading

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus']=False

class MLP(NeuralNetMLP):
    def text_output_func(self, string):
        #self.text_output.configure(state=tk.NORMAL)
        self.text_output.insert('insert',string)
        self.text_output.see('end')
        self.output_ui.update()
        #self.text_output.configure(state=tk.DISABLED)
        return 

# class MLPThread(threading.Thread):
#     def __init__(self, name, X_train, y_train, n_hidden, epochs, eta, shuffle, minibatch_size, seed, text_output= None):#, text_output):

#         self.random = seed
#         self.n_hidden = n_hidden
#         self.epochs = epochs
#         self.eta = eta
#         self.shuffle = shuffle
#         self.minibatch_size = minibatch_size
#         self.text_output = text_output
#         self.result = None

#         self.X_train = X_train
#         self.y_train = y_train

#         self.mlp = MLP( n_hidden=self.n_hidden,
#                  epochs=self.epochs, eta=self.eta,
#                  shuffle=self.shuffle, minibatch_size=self.minibatch_size, seed=self.random, text_output= None)
#         super().__init__(name=name)

#     def run(self):
#         self.result = self.mlp.fit(self.X_train, self.y_train)

#     def get_MLP(self):
#         return self.mlp

class Application():

    def __init__(self):

        # 介面參數
        self.font_name = 'microsoft jhenghei'
        self.filename = ''      # 開啟文件路徑
        self.buffer = list()    # 暫存開啟文件內容 (經處理)
        self.X_train = None     # 訓練集
        self.y_train = None     # 訓練集 (標籤)
        self.X_test = None      # 測試集
        self.y_test = None      # 測試集 (標籤)
        self.weight = None      # 鍵結值
        self.pred_class = None  #        
        self.dim = 0            # 資料維度
        self.mlp = None         # MLP 模型
        self.mlp_result = None  # MLP 模型訓練結果
        self.X = None
        self.X_normal = None
        self.Y = None
        self.normal_flag = False
        self.cost = None
        self.acc = None
        self.cost_trigger = None

        # 模型參數
        self.random = None
        self.n_hidden = None
        self.epochs = None
        self.eta = None
        self.shuffle = None
        self.minibatch_size = None

        self.root = tk.Tk()


        # Row 0
        self.lb_1 = tk.Label(self.root, text="Multi-layer Perceptron Classifier" , font=(self.font_name, 18))#, width="30", height="5")
        self.lb_1.grid(row=0,column=0, columnspan=2, pady=5)

        # Row 1
        self.frame_open = tk.Frame(self.root)
        self.lb_0 = tk.Label(self.frame_open, text="開啟文件" , font=(self.font_name, 10))#, width="30", height="5")
        self.lb_0.grid(row=0,column=0)
        self.bt_open = tk.Button(self.frame_open, text='開啟', font=(self.font_name, 10), command=self.open_file)
        self.bt_open.grid(row=0, column=1)
        self.frame_open.grid(row=1, column=0)

        # self.lb_5 = tk.Label(self.root, text="測試文件" , font=(self.font_name, 9))#, width="30", height="5")
        # self.lb_5.grid(row=1,column=1)
        # self.bt_3 = tk.Button(self.root, text='開啟', font=(self.font_name, 9), command=lambda:self.open_file(type='test'))
        # self.bt_3.grid(row=2, column=1)

        # Row 2
        self.frame_train_set = tk.Frame(self.root)
        self.lb_train_set = tk.Label(self.frame_train_set, text="訓練設定----" , font=(self.font_name, 10))#, width="30", height="5"
        
        self.lb_train_set.grid(row=0,column=0)
        self.lb_eta = tk.Label(self.frame_train_set, text="學習率" , font=(self.font_name, 10))#, width="30", height="5")
        self.lb_eta.grid(row=1,column=0)
        self.entry_eta = tk.Entry(self.frame_train_set ,width='10')
        self.entry_eta.insert(0,'0.1')
        self.entry_eta.grid(row=1, column=1)
        
        self.lb_mbatch = tk.Label(self.frame_train_set, text="最小批次數" , font=(self.font_name, 10))#, width="30", height="5")
        self.lb_mbatch.grid(row=2,column=0)
        self.entry_mbatch = tk.Entry(self.frame_train_set ,width='10')
        self.entry_mbatch.insert(0,'1')
        self.entry_mbatch.grid(row=2, column=1)

        self.lb_hidden = tk.Label(self.frame_train_set, text="隱藏層神經元數" , font=(self.font_name, 10))#, width="30", height="5")
        self.lb_hidden.grid(row=3,column=0)
        self.entry_hidden = tk.Entry(self.frame_train_set ,width='10')
        self.entry_hidden.insert(0,'50')
        self.entry_hidden.grid(row=3, column=1)

        self.chkV_shuffle = tk.BooleanVar()
        self.chkV_shuffle.set(True)
        self.chk_shuffle = tk.Checkbutton(self.frame_train_set, text="是否打亂資料" , var=self.chkV_shuffle ,font=(self.font_name, 10))
        self.chk_shuffle.grid(row=4,column=0, columnspan=2)

        self.chkV_normal = tk.BooleanVar()
        self.chkV_normal.set(False)
        self.chk_normal = tk.Checkbutton(self.frame_train_set, text="是否做資料正規化" , var=self.chkV_normal ,font=(self.font_name, 10))
        self.chk_normal.grid(row=5,column=0, columnspan=2)

        self.frame_train_set.grid(row=2, column=0)

        # Row 3
        self.frame_set = tk.Frame(self.root)
        self.lb_set = tk.Label(self.frame_set, text="收斂條件----" , font=(self.font_name, 10))#, width="30", height="5")
        self.lb_set.grid(row=0,column=0)
        # self.chk_e = tk.Checkbutton(self.frame_set, text='學習循環次數', state=tk.DISABLED) 
        # self.chk_e.select()
        # self.chk_e.grid(row=1, column=0)
        
        self.lb_epoch = tk.Label(self.frame_set, text="學習循環次數" , font=(self.font_name, 10))
        self.lb_epoch.grid(row=1,column=0)
        self.entry_epoch = tk.Entry(self.frame_set ,width='10')
        self.entry_epoch.insert(0,'100')
        self.entry_epoch.grid(row=1, column=1)

        self.chkV_cost = tk.BooleanVar()
        self.chkV_cost.set(False)
        self.chk_cost = tk.Checkbutton(self.frame_set, text="成本限制" ,var=self.chkV_cost , font=(self.font_name, 10))
        self.chk_cost.grid(row=2,column=0)
        self.entry_cost = tk.Entry(self.frame_set ,width='10')
        self.entry_cost.grid(row=2, column=1)

        self.chkV_acc = tk.BooleanVar()
        self.chkV_acc.set(False)
        self.chk_acc = tk.Checkbutton(self.frame_set, text="訓練辨識率限制" ,var=self.chkV_acc , font=(self.font_name, 10))
        self.chk_acc.grid(row=3,column=0)
        self.entry_acc = tk.Entry(self.frame_set ,width='10')
        self.entry_acc.grid(row=3, column=1)

        self.chkV_cost_t = tk.BooleanVar()
        self.chkV_cost_t.set(False)
        self.chk_cost_t = tk.Checkbutton(self.frame_set, text="成本不可大於先前次數" ,var=self.chkV_cost_t , font=(self.font_name, 10))
        self.chk_cost_t.grid(row=4,column=0)
        self.entry_cost_t = tk.Entry(self.frame_set ,width='10')
        self.entry_cost_t.grid(row=4, column=1)

        self.frame_set.grid(row=2, column=1)


        #self.cbt_= tk.Checkbutton(self.frame_set)
        # self.optionList = ("學習循環次數", "誤差限制")
        # self.v = tk.StringVar()
        # self.v.set("選項")
        # self.optionmenu = tk.OptionMenu(self.frame_set, self.v, *self.optionList)
        # self.optionmenu.grid(row=0, column=1)


        # Row 4
        self.lb_3 = tk.Label(self.root, text="輸出結果" , font=(self.font_name, 10))#, width="30", height="5")
        self.lb_3.grid(row=7,column=0)
        self.textframe = tk.Frame(self.root)
        # self.textframe["height"] = 10
        # self.textframe["width"] = 30
        self.scrollbar = tk.Scrollbar(self.textframe)
        self.scrollbar.pack(side=tk.RIGHT,fill=tk.Y)
        self.text = tk.Text(self.textframe, yscrollcommand=self.scrollbar.set)
        self.text["height"] = 10
        self.text["width"] = 52
        self.text.insert('1.0', "初始化，請開啟文件")
        self.text.pack(side=tk.LEFT,fill=tk.BOTH)
        self.scrollbar.config(command=self.text.yview)
        self.text.see('end')
        self.textframe.grid(row=8, column=0, padx=5, columnspan=2)

        # Row 5
        self.bt_train = tk.Button(self.root, text='開始訓練', bg='red', fg='white', font=(self.font_name, 12), command=self.train)
        self.bt_train.grid(row=9, column=0, padx=1)
        self.bt_test = tk.Button(self.root, text='開始測試', bg='red', fg='white', font=(self.font_name, 12), command=self.test)
        self.bt_test.grid(row=9, column=1, padx=1)

        # Row 5
        self.lb_train_acc = tk.Label(self.root, text='訓練辨識率: N/A', font=(self.font_name, 10))
        self.lb_train_acc.grid(row=10, column=0, padx=1)
        self.lb_test_acc = tk.Label(self.root, text='測試辨識率: N/A', font=(self.font_name, 10))
        self.lb_test_acc.grid(row=10, column=1, padx=1)

        # Row 6
        self.lb_plot = tk.Label(self.root, text="繪圖演示" , font=(self.font_name, 12))#, width="30", height="5")
        self.lb_plot.grid(row=0,column=2)
        self.frame_plot = tk.Frame(self.root)
        #self.canvas_true=tk.Canvas(self.frame_true) #創建一塊顯示圖形的畫布
        self.f, self.subf_true, self.subf_pred, self.subf_result = self.create_matplotlib() #返回matplotlib所畫圖形的figure對象
        self.fcanvas_plot, self.toolbar_plot = self.create_form(self.frame_plot, self.f) #將figure顯示在tkinter窗體上面
        self.frame_plot.grid(row=1, column=2, rowspan=10 , padx=5, pady=5)


        # # Row 7
        # self.lb_pred = tk.Label(self.root, text="繪圖演示 (分類結果)" , font=(self.font_name, 10))#, width="30", height="5")
        # self.lb_pred.grid(row=1,column=3)
        # self.frame_pred = tk.Frame(self.root)
        # self.f_pred ,self.subf_pred = self.create_matplotlib() #返回matplotlib所畫圖形的figure對象
        # self.fcanvas_pred, self.toolbar_pred = self.create_form(self.frame_pred, self.f_pred) #將figure顯示在tkinter窗體上面
        # self.frame_pred.grid(row=2, column=3, rowspan=10 , padx=5, pady=5)

        self.root.mainloop()


    def set_true_matplotlib(self):

        #創建繪圖對象f
        #self.f=plt.figure(num=2,figsize=(6,5),dpi=60,facecolor=None,edgecolor='black',frameon=True)
        #self.f.clf()
        #創建一副子圖
        #self.fig1=plt.subplot(1,1,1)

        #fig.title.set_text('預期結果')

        if self.dim > 3:
            return

        self.subf_true.remove()

        if self.dim == 3:
            self.subf_true = self.f.add_subplot(1, 3, 1, projection='3d')
        elif self.dim == 2:
            self.subf_true = self.f.add_subplot(1, 3, 1)
        else:
            return

        self.subf_true.title.set_text('預期結果')
        self.subf_true.set_ylabel('y')
        self.subf_true.set_xlabel('x')

        # temp = self.temp_np
        color_map = ['b.','g.','r.','c.','m.','y.','k.']

        colors = ('red', 'blue', 'green', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(self.y))])

        # temp_x = temp[:,:-1]
        # temp_y = temp[:,-1:]
        # temp_y = temp_y - np.min(temp_y)

        # color_map2 = ["orange","pink","blue","brown","red","grey","yellow","green"]

        # class_ = dict()
        # for i in set(self.temp_np[:,-1:].ravel().tolist()):
        #     class_[int(i)] = np.array([])
        # for i,j in enumerate(temp_y.ravel()):
        #     class_[int(j)] = np.hstack([class_[int(j)],temp_x[i,:]])
        # print(class_)
        #for i in set(self.temp_np[:,-1:].ravel().tolist()):

        # fig1.scatter(temp_x[:,:-1].ravel(),temp_x[:,-1:].ravel(), color_map=color_map2)

        if self.dim == 3:
            self.subf_true.set_zlabel('z')
            for i,s in enumerate(self.X_normal):
                self.subf_true.plot(s[:1],s[1:-1],s[-1:], color_map[int(self.y[i])], markersize=3)#, label='Case '+str(int(temp_y[i])),)
        else:
            # for i,s in enumerate(self.X_normal):
            #     self.subf_true.plot(s[:-1],s[-1:], color_map[int(self.y[i])], markersize=3)#, label='Case '+str(int(temp_y[i])),)
            self.subf_true.scatter(self.X[:,0],self.X[:,1], c=self.y, s=2, cmap=cmap)

        self.toolbar_plot.update()
        self.fcanvas_plot.draw()

    def set_pred_matplotlib(self):
        
        if self.dim > 3:
            return

        self.subf_pred.remove()

        if self.dim == 3:
            self.subf_pred = self.f.add_subplot(1, 3, 2, projection='3d')
        elif self.dim == 2:
            self.subf_pred = self.f.add_subplot(1, 3, 2)
            x0_min, x0_max = self.X_normal[:, 0].min() - 1, self.X_normal[:, 0].max() + 1
            x1_min, x1_max = self.X_normal[:, 1].min() - 1, self.X_normal[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x0_min, x0_max, 0.1), np.arange(x1_min, x1_max, 0.1))
            pred_ALL = self.mlp.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
            print(pred_ALL.shape)
            print(xx, yy)

        self.subf_pred.title.set_text('訓練結果')
        self.subf_pred.set_ylabel('y')
        self.subf_pred.set_xlabel('x')

        # temp = self.temp_np
        color_map = ['b.','g.','r.','c.','m.','y.','k.']
        colors = ListedColormap(['yello','blue','green','red'])

        # temp_x = temp[:,:-1]
        # temp_y = temp[:,-1:] 
        #print(temp)
        #temp_y = temp_y - np.min(temp_y)

        pred_y = self.mlp.predict(self.X_normal)
        colors = ('red', 'blue', 'green', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(pred_y))])

        #print(pred_y)
        #pred_y = pred_y - np.min(pred_y)

        if self.dim == 3:
            self.subf_pred.set_zlabel('z')
            for i,s in enumerate(self.X):
                self.subf_pred.plot(s[:1],s[1:-1],s[-1:], color_map[int(pred_y[i])], markersize=3)#, label='Case '+str(int(temp_y[i])),)
        else:
            #for i,s in enumerate(self.X):
                #self.subf_pred.plot(s[:-1],s[-1:], color_map[int(pred_y[i])], markersize=3)#, label='Case '+str(int(temp_y[i])),)
            self.subf_pred.scatter(self.X[:,0],self.X[:,1], c=pred_y, s=2, cmap=cmap)
            self.subf_pred.contourf(xx, yy, pred_ALL, cmap=cmap, alpha=0.4)#, cmap=colors)
                
        #self.toolbar_plot.update()
        #self.fcanvas_plot.draw()

    def set_result_matplotlib(self):

        self.subf_result.remove()

        self.subf_result = self.f.add_subplot(1, 3, 3)
        self.subf_result.title.set_text('訓練紀錄')

        train_acc_100 = [i*100 for i in self.mlp.eval_['train_acc']]
        self.subf_result.plot(range(len(self.mlp.eval_['cost'])), self.mlp.eval_['cost'], label='cost')
        self.subf_result.plot(range(len(train_acc_100)), train_acc_100, label='training acc. (100%)')

        self.subf_result.set_ylabel('y')
        self.subf_result.set_xlabel('x')
        self.subf_result.legend()

        self.toolbar_plot.update()
        self.fcanvas_plot.draw()
        
        # fig.plot(range(self.epochs), self.mlp.eval_['train_acc'], label='training')
        # fig.set_ylabel('Accuracy')
        # fig.set_xlabel('Epochs')
        # fig.legend()

    def create_form(self, frame, figure):
        #把繪製的圖形顯示到tkinter窗口上
        fcanvas=FigureCanvasTkAgg(figure, frame)
        fcanvas.draw()  #以前的版本使用show()方法，matplotlib 2.2之後不再推薦show（）用draw代替，但是用show不會報錯，會顯示警告
        fcanvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        #把matplotlib繪製圖形的導航工具欄顯示到tkinter窗口上
        toolbar =NavigationToolbar2Tk(fcanvas, frame) #matplotlib 2.2版本之後推薦使用NavigationToolbar2Tk，若使用NavigationToolbar2TkAgg會警告
        toolbar.update()
        fcanvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        return fcanvas, toolbar

    def create_matplotlib(self):
        #創建繪圖對象f
        f = Figure(figsize=(12,4), dpi=80) #,facecolor=None,edgecolor='black',frameon=True)
        #f = Figure(figsize=(9,3), dpi=100) #,facecolor=None,edgecolor='black',frameon=True)
        #創建一副子圖
        subf_1 = f.add_subplot(1,3,1)
        subf_1.title.set_text('預期結果')
        subf_2 = f.add_subplot(1,3,2)
        subf_2.title.set_text('訓練結果')
        subf_3 = f.add_subplot(1,3,3)
        subf_3.title.set_text('訓練紀錄')
        return f, subf_1, subf_2, subf_3

    def init(self):
        self.filename = ''      # 開啟文件路徑
        self.buffer = list()    # 暫存開啟文件內容 (經處理)
        self.X_train = None     # 訓練集
        self.y_train = None     # 訓練集 (標籤)
        self.X_test = None      # 測試集
        self.y_test = None      # 測試集 (標籤)
        self.weight = None      # 鍵結值
        self.pred_class = None  #        
        self.dim = 0            # 資料維度
        self.mlp = None         # MLP 模型
        self.mlp_result = None  # MLP 模型訓練結果
        self.X = None
        self.X_normal = None
        self.Y = None
        self.normal_flag = False
        self.cost = None
        self.acc = None
        self.cost_trigger = None
        
        # 模型參數
        self.random = None
        self.n_hidden = None
        self.epochs = None
        self.eta = None
        self.shuffle = None
        self.minibatch_size = None

        self.text.delete(1.0,'end')
        self.text.insert('1.0', "初始化，請開啟文件")

        # 清除畫布
        # #plt.close(self.f)
        # #self.f = Figure(figsize=(12,4),dpi=80)
        # #self.subf_true.clear()
        # self.subf_true.remove()
        # self.subf_true = self.f.add_subplot(1,3,1)
        # self.subf_true.title.set_text('預期結果')
        # self.subf_pred.remove()
        # self.subf_pred = self.f.add_subplot(1,3,2)
        # self.subf_pred.title.set_text('訓練結果')
        # self.subf_result.remove()
        # self.subf_result = self.f.add_subplot(1,3,3)
        # self.subf_result.title.set_text('訓練紀錄')
        # # self.fcanvas_plot.figure = self.f
        # # self.toolbar_plot.canvas = self.fcanvas_plot
        # # self.toolbar_plot.figure = self.f
        # self.toolbar_plot.update()
        # self.fcanvas_plot.draw()

        # 重置畫布 Frame
        self.frame_plot.destroy()
        self.frame_plot = tk.Frame(self.root)
        self.f, self.subf_true, self.subf_pred, self.subf_result = self.create_matplotlib() #返回matplotlib所畫圖形的figure對象
        self.fcanvas_plot, self.toolbar_plot = self.create_form(self.frame_plot, self.f) #將figure顯示在tkinter窗體上面
        self.frame_plot.grid(row=1, column=2, rowspan=10 , padx=5, pady=5)

        self.bt_train['text'] = '開始訓練'
        self.bt_test['text'] = '開始測試'
        self.lb_train_acc['text']='訓練辨識率: N/A'
        self.lb_test_acc['text']='測試辨識率: N/A'

    def open_file(self):
        self.init()
        self.text.insert('insert','\n----打開文件----')
        self.filename = fd.askopenfilename(filetypes = (("Text files", "*.txt"),("All files", "*.*") ))
        if self.filename == '':
            self.text.insert('insert','\n----開啟失敗----')
            return
        
        # 讀取文件內容
        with open(self.filename) as file:
            for i in file.readlines():
                self.buffer.append([float(i) for i in i.rstrip().split(' ')])   # str->float
                #print(i.rstrip().split(' '))
        self.text.insert('insert','\n資料路徑: '+str(self.filename))

        self.dim = len(self.buffer[0]) - 1
        self.text.insert('insert','\n資料維度: '+str(self.dim))

        if self.dim > 3:
            showinfo('注意','目前的資料集維度 > 3，將無法畫圖')
        #print('訓練的資料維度:',self.dim)

        self.data_pre_processing(True)

        # # 初始化參數 (附上 bias)
        # self.weight = np.random.randn(1,self.dim)
        # self.weight = np.hstack([np.array(self.theta).reshape(-1,1),self.weight])
        # print(self.weight)

        # # 設定 X_0 = -1
        # self.X_train = np.hstack([np.repeat(-1,train_size).reshape(train_size,1),self.X_train])
        # #print(self.X_train)
        # self.X_test = np.hstack([np.repeat(-1,test_size).reshape(test_size,1),self.X_test])
        # #print(self.X_test)

        # print(int(np.max(self.temp_np[:,-1:])))
        # print(np.arange(0,int(np.max(self.temp_np[:,-1:]))))
        # print(np.unique(np.arange(0,np.unique(self.temp_np[:,-1:]).shape[0])))

        # self.y_train = self.y_train.ravel()-np.min(self.temp_np[:,-1:])
        # self.y_test = self.y_test.ravel()-np.min(self.temp_np[:,-1:])
        # print(self.y_train.ravel()-np.min(self.y_train))
        # print(self.y_train.shape)
        # print(self._onehot(self.y_train.ravel()-np.min(self.y_train) ,self.pred_class))

        #if self.dim == 2:
        self.set_true_matplotlib()

    def popup_hello(self):
        showinfo("Hello", "Hello Tk!")
    
    def data_pre_processing(self, info_output):
        # 處理資料
        self.temp_np = np.array(self.buffer)
        #self.temp_np = temp_np
        np.random.shuffle(self.temp_np) # 打亂
        data_size = len(self.buffer)
        self.text.insert('insert','\n資料大小: '+str(data_size))

        train_size = round(data_size*(2.0/3.0))
        test_size = data_size - train_size

        self.X = self.temp_np[:,:-1]
        self.y = self.temp_np[:,-1:]
        self.y = self.y.ravel()-np.min(self.y)

        if self.normal_flag:
            self.X_normal = np.array(self.X)
            for i in range(self.X.shape[1]):
                self.X_normal[:,i:i+1] = self.X[:,i:i+1]/np.linalg.norm(self.X[:,i:i+1])
        else:
            self.X_normal = self.X

        self.X_train = self.X_normal[:train_size]
        self.X_test = self.X_normal[train_size:]
        self.y_train = self.y[:train_size]
        self.y_test = self.y[train_size:]

        if info_output:
            self.text.insert('insert','\n訓練集大小: '+str(self.X_train.shape))
            self.text.insert('insert','\n測試集大小: '+str(self.X_test.shape))

            self.pred_class = np.unique(self.temp_np[:,-1:]).shape[0]
            #print(self.y_train)
            self.text.insert('insert','\n分類數: '+str(self.pred_class))

    def train(self):
        
        # 關閉按鈕
        self.bt_open['state'] = tk.DISABLED
        self.bt_train['state'] = tk.DISABLED

        self.text.insert('insert','\n\n----開始訓練----')
        if self.filename == '': 
            showinfo('錯誤','未打開文件或開啟失敗')
            self.bt_train['state'] = tk.NORMAL
            self.text.insert('insert','\n----訓練失敗----')
            return
        
        # 參數讀取 (訓練參數)
        try:
            self.eta = float(self.entry_eta.get())
            self.minibatch_size = int(self.entry_mbatch.get())
            self.n_hidden = int(self.entry_hidden.get())
            self.shuffle = self.chkV_shuffle.get()
            self.normal_flag = self.chkV_normal.get()
            if (self.eta <= 0.0) or (self.minibatch_size  <= 0 or (self.n_hidden <= 0)):
                raise ValueError('ValueError: ','eta ,minibatch_size or n_hidden 參數錯誤')
        except Exception as e:
            showinfo('錯誤','訓練參數設定錯誤，\n請注意輸入格式！\n\n說明:\n\n  學習率環次數應大於 0.0\n  最小批次數應大於等於 1 且為整數\n  隱藏層神經元數應大於 1 且為整數')
            self.bt_train['state'] = tk.NORMAL
            self.text2out('\n----訓練失敗----')
            self.bt_open['state'] = tk.NORMAL
            print('[!]錯誤訊息: ',e)
            return

        # 參數讀取 (收斂條件)
        try:
            self.epochs = int(self.entry_epoch.get())

            if self.chkV_cost.get():
                self.cost = float(self.entry_cost.get())
                if (self.epochs < 1):
                    raise ValueError('ValueError: ','學習循環次數 參數錯誤')
            else:
                self.cost = None

            if self.chkV_acc.get():
                self.acc = float(self.entry_acc.get())
                if (self.acc < 0.0) or (self.acc > 1.0):
                    raise ValueError('ValueError: ','訓練辨識率限制 參數錯誤')
            else:
                self.acc = None

            if self.chkV_cost_t.get():
                self.cost_trigger = int(self.entry_cost_t.get())
                if (self.cost_trigger < 1):
                    raise ValueError('ValueError: ','成本不可大於先前次數 參數錯誤')
            else:
                self.acc = None

        except Exception as e:
            showinfo('錯誤','收斂條件設定錯誤，\n請注意輸入格式！\n\n說明:\n\n  學習循環次數應大於 1 且為整數\n\n  (選)成本限制應為數值\n\n (選)訓練辨識率限制為 0 ~ 1 之間的浮點數\n\n  (選)成本不可大於先前次數必為整數且大於等於1')
            self.bt_train['state'] = tk.NORMAL
            self.text2out('\n----訓練失敗----')
            self.bt_open['state'] = tk.NORMAL
            print('[!]錯誤訊息: ',e)
            return   
        
        if self.normal_flag:
            self.text2out('\n偵測到正規化勾選，重新整理資料...')
            self.data_pre_processing(True)
        else:
            self.data_pre_processing(False)

        self.text2out('\n學習率: '+str(self.eta))
        self.text2out('\n最小批次數: '+str(self.minibatch_size))
        self.text2out('\n隱藏層神經元數: '+str(self.n_hidden))
        self.text2out('\n打亂資料: '+str(self.shuffle))
        self.text2out('\n學習循環次數: '+str(self.epochs))

        # 建立模型
        self.text2out('\n****模型輸出****')
        self.mlp = MLP( n_hidden=self.n_hidden,
                 epochs=self.epochs, eta=self.eta,
                 shuffle=self.shuffle, minibatch_size=self.minibatch_size, seed=None, 
                 text_output= self.text, output_ui=self.root, cost_limit=self.cost, acc_limit=self.acc,
                 cost_trigger=self.cost_trigger)
        self.mlp_result = self.mlp.fit(X_train=self.X_train, y_train=self.y_train)
        self.text2out('\n****輸出結束****')
        # thread = MLPThread('MLP_thread', X_train=self.X_train, y_train=self.y_train, n_hidden=self.n_hidden,
        #           epochs=100, eta=self.eta,
        #           shuffle=self.shuffle, minibatch_size=self.minibatch_size, seed=None, text_output= self.text)
        # thread.start()
        # thread.join()
        # self.mlp =  thread.get_MLP()

        y_train_pred = self.mlp.predict(self.X_train)
        self.text.insert('insert','\n----訓練成功----')

        acc = (np.sum(self.y_train == y_train_pred).astype(np.float64) / self.X_train.shape[0])
        print('\nTraining accuracy: %.2f%%'%(acc*100))
        self.text2out('\n----訓練結果----')
        self.text2out('\n訓練辨識率: %.2f%%'%(acc*100))

        self.text2out('\n模型內部參數:')
        self.text2out('\ninput layer ---------> hidden layer')
        self.text2out('\n鍵結值大小: '+str(self.mlp.w_h.shape))
        self.text2out('\n鍵結值內容: \n'+str(self.mlp.w_h))
        self.text2out('\n閥值大小: '+str(self.mlp.b_h.shape))
        self.text2out('\n閥值內容: \n'+str(self.mlp.b_h))

        self.text2out('\nhidden layer ---------> output layer')
        self.text2out('\n鍵結值大小: '+str(self.mlp.w_out.shape))
        self.text2out('\n鍵結值內容: \n'+str(self.mlp.w_out))
        self.text2out('\n閥值大小: '+str(self.mlp.b_out.shape))
        self.text2out('\n閥值內容: \n'+str(self.mlp.b_out))

        # 結束

        self.set_pred_matplotlib()      # 畫訓練結果圖
        self.set_result_matplotlib()    # 畫訓練記錄圖

        self.lb_train_acc['text'] = '訓練辨識率: %.2f%%'%(acc*100)
        self.bt_open['state'] = tk.NORMAL
        self.bt_train['state'] = tk.NORMAL
        self.bt_train['text'] = '重新訓練'

        self.test()


    def test(self):

        # 關閉按鈕
        self.bt_open['state'] = tk.DISABLED
        self.bt_test['state'] = tk.DISABLED

        self.text2out('\n\n----開始測試----')
        if self.filename == '': 
            showinfo('錯誤','未打開文件')
            self.text2out('\n----測試失敗----')
            self.bt_test['state'] = tk.NORMAL
            return

        if self.mlp == None:
            showinfo('錯誤','未曾訓練過')
            self.text2out('\n----測試失敗----')
            self.bt_test['state'] = tk.NORMAL
            return  

        y_test_pred = self.mlp.predict(self.X_test)
        self.text2out('\n----測試成功----')

        acc = (np.sum(self.y_test == y_test_pred).astype(np.float64) / self.X_test.shape[0])
        print('Testing accuracy: %.2f%%'%(acc*100))
        self.text2out('\n----測試結果----')
        self.text2out('\n測試辨識率: %.2f%%'%(acc*100))
        self.text.see('end')

        self.lb_test_acc['text'] = '測試辨識率: %.2f%%'%(acc*100)

        # 結束
        self.bt_open['state'] = tk.NORMAL
        self.bt_test['state'] = tk.NORMAL
        self.bt_test['text'] = '重新測試'

    def text2out(self, string):
        #self.text.configure(state=tk.NORMAL)
        self.text.insert('insert',string)
        self.text.see('end')
        self.root.update()
        #self.text.configure(state=tk.DISABLED)
        return 

    # def create_button(self,txt):
    #     bt_1 = tk.Button(root, text=str(txt), bg='red', fg='white', font=('Arial', 12))
    #     bt_1['width'] = 50
    #     bt_1['height'] = 4
    #     #bt_1['command'] = set_text
    #     bt_1['activebackground'] = 'blue' 
    #     bt_1['activeforeground'] = 'yellow'
    #     bt_1.pack()

app = Application()

