import tkinter as tk
from tkinter.messagebox import showinfo
from tkinter import filedialog as fd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk #NavigationToolbar2TkAg
import numpy as np
from numpy.core.fromnumeric import size

class Application():

    def __init__(self):
        self.font_name = 'microsoft jhenghei ui'
        self.filename = ''      # 開啟文件路徑
        self.buffer = list()    # 暫存開啟文件內容 (經處理)
        self.X_train = None     # 訓練集
        self.y_train = None     # 訓練集 (標籤)
        self.X_test = None      # 測試集
        self.y_test = None      # 測試集 (標籤)
        self.weight = None      # 鍵結值
        self.theta = -1         # 神經元伐值
        self.pred_class = None  #        
        self.n = 1              # 學習循環次數
        self.dim = 0            # 資料維度
        self.root = tk.Tk()

        # Row 0
        self.lb_1 = tk.Label(self.root, text="Multi-layer Perceptron Classifier" , font=(self.font_name, 18))#, width="30", height="5")
        self.lb_1.grid(row=0,column=0, columnspan=3, pady=5)

        # Row 1
        self.lb_0 = tk.Label(self.root, text="開啟文件" , font=(self.font_name, 9))#, width="30", height="5")
        self.lb_0.grid(row=1,column=0)
        self.bt_1 = tk.Button(self.root, text='開啟', font=(self.font_name, 9), command=self.open_file)
        self.bt_1.grid(row=2, column=0)
        # self.lb_5 = tk.Label(self.root, text="測試文件" , font=(self.font_name, 9))#, width="30", height="5")
        # self.lb_5.grid(row=1,column=1)
        # self.bt_3 = tk.Button(self.root, text='開啟', font=(self.font_name, 9), command=lambda:self.open_file(type='test'))
        # self.bt_3.grid(row=2, column=1)

        # Row 2
        self.lb_2 = tk.Label(self.root, text="學習率" , font=(self.font_name, 9))#, width="30", height="5")
        self.lb_2.grid(row=3,column=0)
        self.entry_1 = tk.Entry(self.root, width='10')
        self.entry_1.grid(row=4, column=0)

        # Row 3
        self.lb_3 = tk.Label(self.root, text="收斂條件" , font=(self.font_name, 9))#, width="30", height="5")
        self.lb_3.grid(row=3,column=1)
        self.cbt_1 = tk.Checkbutton(self.root)
        self.optionList = ("學習循環次數", "誤差限制")
        self.v = tk.StringVar()
        self.v.set("選項")
        self.optionmenu = tk.OptionMenu(self.root, self.v, *self.optionList)
        self.optionmenu.grid(row=4, column=1)

        # Row 4
        self.lb_3 = tk.Label(self.root, text="輸出結果" , font=(self.font_name, 9))#, width="30", height="5")
        self.lb_3.grid(row=7,column=0)
        self.text = tk.Text(self.root)
        self.text["height"] = 10
        self.text["width"] = 30
        # # 設定 tag
        # self.text.tag_config("tag_1", backgroun="yellow", foreground="red")
        # "insert" 索引表示插入游標當前的位置
        self.text.insert('1.0', "初始化，請開啟文件")
        #print(self.text.get(1.0,"end"))
        self.text['state'] = 'disabled'
        self.text['state'] = 'normal'
        #self.text.delete(1.0,"end")
        self.text.grid(row=8, column=0, padx=5 , columnspan=2)

        # Row 5
        self.bt_train = tk.Button(self.root, text='開始訓練', bg='red', fg='white', font=(self.font_name, 12), command=self.train)
        self.bt_train.grid(row=9, column=0, padx=1)
        self.bt_test = tk.Button(self.root, text='開始測試', bg='red', fg='white', font=(self.font_name, 12), command=self.test)
        self.bt_test.grid(row=9, column=1, padx=1)

        # Row 6
        self.lb_4 = tk.Label(self.root, text="繪圖演示" , font=(self.font_name, 9))#, width="30", height="5")
        self.lb_4.grid(row=1,column=2)
        self.frame = tk.Frame(self.root)
        self.canvas=tk.Canvas(self.frame) #創建一塊顯示圖形的畫布
        self.create_matplotlib() #返回matplotlib所畫圖形的figure對象
        self.create_form(self.f) #將figure顯示在tkinter窗體上面
        self.frame.grid(row=2,column=2, rowspan=10 , padx=5, pady=5)
        self.root.mainloop()

    def set_matplotlib(self):

        #創建繪圖對象f
        #self.f=plt.figure(num=2,figsize=(6,5),dpi=60,facecolor=None,edgecolor='black',frameon=True)
        #self.f.clf()
        #創建一副子圖
        #self.fig1=plt.subplot(1,1,1)

        temp = self.temp_np
        temp_x = temp[:,:-1]
        temp_y = temp[:,-1:]
        color_map = ['r.','g.','b.','c.','m.','y.','k.']
        # color_map2 = ["orange","pink","blue","brown","red","grey","yellow","green"]

        # class_ = dict()
        # for i in set(self.temp_np[:,-1:].ravel().tolist()):
        #     class_[int(i)] = np.array([])
        # for i,j in enumerate(temp_y.ravel()):
        #     class_[int(j)] = np.hstack([class_[int(j)],temp_x[i,:]])
        # print(class_)
        #for i in set(self.temp_np[:,-1:].ravel().tolist()):

        # fig1.scatter(temp_x[:,:-1].ravel(),temp_x[:,-1:].ravel(), color_map=color_map2)
        for i,s in enumerate(temp_x):
            self.fig1.plot(s[:-1],s[-1:], color_map[int(temp_y[i])], markersize=10)#, label='Case '+str(int(temp_y[i])),)

    def create_form(self,figure):
        #把繪製的圖形顯示到tkinter窗口上
        self.canvas=FigureCanvasTkAgg(figure,self.frame)
        self.canvas.draw()  #以前的版本使用show()方法，matplotlib 2.2之後不再推薦show（）用draw代替，但是用show不會報錯，會顯示警告
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        #把matplotlib繪製圖形的導航工具欄顯示到tkinter窗口上
        toolbar =NavigationToolbar2Tk(self.canvas, self.frame) #matplotlib 2.2版本之後推薦使用NavigationToolbar2Tk，若使用NavigationToolbar2TkAgg會警告
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def create_matplotlib(self):
        #創建繪圖對象f
        self.f=plt.figure(num=2,figsize=(6,5),dpi=60,facecolor=None,edgecolor='black',frameon=True)
        #創建一副子圖
        self.fig1=plt.subplot(1,1,1)

        # x=np.arange(-3,3,0.1)
        # y1=np.sin(x)
        # y2=np.cos(x)

        # line1,=fig1.plot(x,y1,color='red',linewidth=3,linestyle='--')    #畫第一條線
        # line2,=fig1.plot(x,y2,color='blue',linewidth=3,linestyle='-') 
        # #plt.setp(line2,color='black',linewidth=8,linestyle='-',alpha=0.3)#華第二條線

        # fig1.set_title("Picture 1",loc='center',pad=20,fontsize='xx-large',color='red')    #設置標題                                                        #確定圖例
        # fig1.legend(['sin','cos'],loc='upper left',facecolor='green',frameon=True,shadow=True,framealpha=0.5,fontsize='xx-large')

        # fig1.set_xlabel('y')                                                             #確定座標軸標題
        # fig1.set_ylabel("x")
        # fig1.set_yticks([-1,-1/2,0,1/2,1])                                                   #設置座標軸刻度
        # fig1.grid(which='major',axis='x',color='gray', linestyle='-', linewidth=0.5)              #設置網格
        


    def init(self):
        self.filename = ''      # 開啟文件路徑
        self.buffer = list()    # 暫存開啟文件內容 (經處理)
        self.X_train = None     # 訓練集
        self.y_train = None     # 訓練集 (標籤)
        self.X_test = None      # 測試集
        self.y_test = None      # 測試集 (標籤)
        self.weight = None      # 鍵結值
        self.theta = -1         # 神經元伐值
        self.pred_class = None  #        
        self.n = 1              # 學習循環次數
        self.dim = 0            # 資料維度
        self.text.delete(1.0,'end')
        self.text.insert('1.0', "初始化，請開啟文件")

        # 清除畫布
        self.fig1=plt.clf()
        self.fig1=plt.subplot(1,1,1)
        self.canvas.draw()

        self.bt_train['text'] = '開始訓練'
        self.bt_test['text'] = '開始測試'

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
        #print('訓練的資料維度:',self.dim)

        self.temp_np = np.array(self.buffer)
        #self.temp_np = temp_np
        np.random.shuffle(self.temp_np) # 打亂
        data_size = len(self.buffer)
        self.text.insert('insert','\n資料大小: '+str(data_size))

        train_size = round(data_size*(2.0/3.0))
        test_size = data_size - train_size
        self.X_train = self.temp_np[:train_size,:-1]
        self.X_test = self.temp_np[train_size:,:-1]
        self.y_train = self.temp_np[:train_size,-1:]
        self.y_test = self.temp_np[train_size:,-1:]
        self.text.insert('insert','\n訓練集大小: '+str(self.X_train.shape))
        self.text.insert('insert','\n測試集大小: '+str(self.X_test.shape))

        # # 初始化參數 (附上 bias)
        # self.weight = np.random.randn(1,self.dim)
        # self.weight = np.hstack([np.array(self.theta).reshape(-1,1),self.weight])
        # print(self.weight)

        # # 設定 X_0 = -1
        # self.X_train = np.hstack([np.repeat(-1,train_size).reshape(train_size,1),self.X_train])
        # #print(self.X_train)
        # self.X_test = np.hstack([np.repeat(-1,test_size).reshape(test_size,1),self.X_test])
        # #print(self.X_test)

        self.pred_class = np.unique(self.temp_np[:,-1:]).shape[0]
        #print(self.y_train)
        self.text.insert('insert','\n分類數: '+str(self.pred_class))

        self.y_train = self.y_train.ravel()-np.min(self.y_train)
        self.y_test = self.y_test.ravel()-np.min(self.y_test)
        # print(self.y_train.ravel()-np.min(self.y_train))
        # print(self.y_train.shape)
        # print(self._onehot(self.y_train.ravel()-np.min(self.y_train) ,self.pred_class))

        if self.dim == 2:
            self.set_matplotlib()
            self.canvas.draw()

    def popup_hello(self):
        showinfo("Hello", "Hello Tk!")
    
    def train(self):
        self.text.insert('insert','\n----開始訓練----')
        self.bt_train['text'] = '再訓練'
        if self.filename == '': 
            showinfo('錯誤','未打開文件或開啟失敗')
            self.bt_train['text'] = '開始訓練'
            self.text.insert('insert','\n----訓練失敗----')
            return
        
        nn = NeuralNetMLP( n_hidden=30,
                 epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None, oputui= self.text)
        nn.fit(X_train=self.X_train, y_train=self.y_train)


    def test(self):
        self.bt_test['text'] = '再測試'
        if self.filename == '': 
            showinfo('錯誤','未打開文件')
            self.bt_test['text'] = '開始測試'


    # def create_button(self,txt):
    #     bt_1 = tk.Button(root, text=str(txt), bg='red', fg='white', font=('Arial', 12))
    #     bt_1['width'] = 50
    #     bt_1['height'] = 4
    #     #bt_1['command'] = set_text
    #     bt_1['activebackground'] = 'blue' 
    #     bt_1['activeforeground'] = 'yellow'
    #     bt_1.pack()

class NeuralNetMLP(object):
    """

    參數:
    -----------
    n_hidden : int 
        隱藏層的神經元數
    epcohs : int
        學習次數
    eta : float
        學習率
    shuffle : bool
        每個 epoch 中，是否將訓練資料打亂
    minibatch_size : int
        訓練資料的取樣數目
    seed : int
        隨機種子，用來產生初始 weights 和打亂的順序

    """
    def __init__(self, n_hidden=30,
                 epochs=100, eta=0.001,
                 shuffle=True, minibatch_size=1, seed=None, oputui = None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.oputui = oputui

    def _onehot(self, y, n_classes):
        """將 Label 編碼成 one-hot 表示法

        參數
        ------------
        y : array, shape = [樣本數]
            期望值.

        返回
        -----------
        onehot : array, shape = (樣本數, 標籤數)

        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            print(idx, val)
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """計算 logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250))) # 限制 z 最多在 -250~250 之間
    
    def _forward(self, X):
        """計算 forward propagation 步驟"""

        # step 1: net input of hidden layer
        # [n_hidden, n_features] dot [n_features ,n_samples]
        # -> [n_hidden , n_samples]
        z_h = np.dot(self.w_h, X.T)

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)
        
        # step 3: net input of output layer
        # [n_classlabels, n_hidden] dot [n_hidden , n_samples]
        # -> [n_classlabels, n_samples]
        z_out = np.dot(self.w_out, a_h)

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out

    def _compute_cost(self, y_enc, output):

        cost = np.sum((y_enc - output)**2)
        return cost
    
    def fit(self, X_train, y_train):
        n_output = np.unique(y_train).shape[0]  # number of class labels
        n_features = X_train.shape[1]

        self.oputui.insert('insert','\n神經輸出: '+str(n_output)+'\n輸入特徵: '+str(n_features))
        print(n_output, n_features)

app = Application()

