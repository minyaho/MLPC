import numpy as np
import sys

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
    text_output : None
        是否設定文字資訊輸出，請自行改寫 self.text_output_func()
    output_ui: None
        文字資訊輸出的UI，可以自行考慮是否要用，並改寫 self.text_output_func() 來呼叫
    cost_limit: None
        限制最低花費，若無輸入則無限制
     acc_limit: None
        限制最高訓練辨識率，若無輸入則無限制   
     cost_trigger: None
        限制成本回彈的次數(本次成本不能大於上次)，若無輸入則無限制    
    """
    def __init__(self, n_hidden=30,
                 epochs=100, eta=0.1,
                 shuffle=True, minibatch_size=1, seed=None, text_output = None, output_ui=None, cost_limit=None, acc_limit=None, cost_trigger=None):
        self.random = np.random.RandomState(seed)
        self.n_hidden = n_hidden
        self.epochs = epochs
        self.eta = eta
        self.shuffle = shuffle
        self.minibatch_size = minibatch_size
        self.text_output = text_output
        self.output_ui = output_ui
        self.cost_limit = cost_limit
        self.acc_limit = acc_limit
        self.cost_trigger = cost_trigger
        self.cost_count = 0                 # 成本回彈的次數的計數器

        # 偵測有無設立出書介面，若有則會開啟輸出
        # 輸出時會呼叫 self.text_output_func(str)
        # 需自行改寫 self.text_output_func()
        if self.text_output == None:        
            self.out_Text_flag = False
        else:
            self.out_Text_flag = True

    def _onehot(self, y, n_classes):
        """
        將 Label 編碼成 one-hot 表示法

        參數
        ------------
        y : array, shape = [樣本數]
            期望值.

        返回
        -----------
        onehot : array, shape = (樣本數, 標籤數)
            one-hot array
        """
        onehot = np.zeros((n_classes, y.shape[0]))
        for idx, val in enumerate(y.astype(int)):
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """
        計算 logistic function (sigmoid)
        
        參數
        ------------
        z : array, shape = (n,m) or value
            輸入計算

        返回
        -----------
        z_s : array, shape = (n,m) or value
            計算結果
        """
        return 1. / (1. + np.exp(-np.clip(z, -250, 250))) # 限制 z 最多在 -250~250 之間
    
    def _forward(self, X):
        """
        計算 forward propagation 步驟

        參數
        ------------
        X : array, shape = (樣本數,特徵數)
            forward 的輸入 

        返回
        -----------
        z_h : array, shape = (樣本數,隱藏層神經元數)
            未經活化的隱藏層輸出
        a_h : array, shape = (樣本數,隱藏層神經元數)
            經活化的隱藏層輸出
        z_out : array, shape = (樣本數,分類數)
            未經活化的輸出層輸出
        a_out : array, shape = (樣本數,分類數)
            經活化的輸出層輸出
        """

        # step 1: net input of hidden layer
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        z_h = np.dot(X, self.w_h) + self.b_h

        # step 2: activation of hidden layer
        a_h = self._sigmoid(z_h)

        # step 3: net input of output layer
        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]

        z_out = np.dot(a_h, self.w_out) + self.b_out

        # step 4: activation output layer
        a_out = self._sigmoid(z_out)

        return z_h, a_h, z_out, a_out


    def _compute_cost(self, y_enc, output):
        """
        成本計算

        參數
        ------------
        y_enc : array, shape = (樣本數, 標籤數)
            預期輸出
        output : array, shape = (樣本數, 標籤數)
            模型輸出

        返回
        -----------
        cost : float
            成本
        """
        cost = np.sum((y_enc - output)**2)/2.0
        return cost
    
    def fit(self, X_train, y_train):
        """
        產生適合網路訓練的資訊與資料

        參數
        ------------
        X_train : array, shape = (樣本數, 特徵數)
            訓練資料與其特徵
        y_train : array, shape = (樣本數, 1)
            訓練資料與其標籤，注意: 第 1 維是標籤

        返回
        -----------
        self : NeuralNetMLP
            模型本身
        """

        # 計算欲分類數
        # !!!!!
        # (Due to the diversity of the dataset, we need to add 1 class to represent 0 bits.)
        # if you think about that will influence the result, you can remove it, but you need check your dataset before you do.
        # !!!!!
        n_output = np.unique(y_train).shape[0] + 1 # number of class labels 

        # 計算特徵數
        n_features = X_train.shape[1]

        if self.out_Text_flag:  # 輸出結果至 UI 介面
            self.text_output_func('\n神經輸出: '+str(n_output-1)+'\n輸入特徵: '+str(n_features))

        ########################
        # Weight initialization
        ########################

        # weights for input -> hidde
        #   b_h: 隱藏層閥值
        #   w_h: 隱藏層鍵結值
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,size=(n_features, self.n_hidden))

        # weights for hidden -> output
        #   b_out: 隱藏層閥值
        #   w_out: 隱藏層鍵結值
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,size=(self.n_hidden, n_output))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        
        # 設立訓練歷史紀錄庫
        self.eval_ = {'cost': [], 'train_acc': []}

        # 將標籤轉換成 one-hot 形式
        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            # 是否打亂資料
            if self.shuffle:
                self.random.shuffle(indices)

            # 產生批次數大小的資料，餵入 forward propagation，並修正鍵結值與閥值
            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):

                # 提取出本梯次會使用到的索引編號
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # 從X_train中拉出，放入 forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # 計算 Error value
                # [n_samples, n_classlabels]
                e_out = a_out - y_train_enc[batch_idx]

                # 計算輸出層 Cost function 的倒數
                # [n_samples, n_classlabels]
                sigmoid_derivative_o = a_out * (1. - a_out)

                # 計算隱藏層 Cost function 的倒數
                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # 計算輸出層的修正變數 detla 
                # [n_samples, n_classlabels] 
                detla_o = e_out * sigmoid_derivative_o

                # 計算隱藏層的修正變數 detla
                # [n_samples, n_hidden] = [n_samples, n_hidden] * [n_samples, n_classlabels]  * [n_hidden, n_classlabels]
                delta_h = sigmoid_derivative_h * np.dot(detla_o, self.w_out.T)

                # 修正隱藏層鍵結值與閥值
                # [n_samples, n_hidden] = [n_samples, n_hidden] - eta * [n_samples, n_features].T dot [n_samples, n_hidden]
                self.w_h -= self.eta * np.dot(X_train[batch_idx].T, delta_h) 
                self.b_h -= self.eta * np.sum(delta_h, axis=0)
                
                # 修正輸出層鍵結值與閥值
                # [n_hidden, n_classlabels] = [n_hidden, n_classlabels] - eta * [n_samples, n_hidden].T dot [n_samples, n_classlabels]  
                self.w_out -= self.eta * np.dot(a_h.T, detla_o)
                self.b_out -= self.eta * np.sum(e_out, axis=0)

            #############
            # Evaluation
            #############

            # 每個 epoch 後都會評估模型訓練效果

            # 餵入所有資料到網路內來獲得結果 a_out
            z_h, a_h, z_out, a_out = self._forward(X_train)

            # 計算與預期結果之間的誤差成本
            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)

            # 讓模型對所有資料做預測
            y_train_pred = self.predict(X_train)

            # 計算預測的辨識率
            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float64) /
                         X_train.shape[0])

            # 存入訓練歷史
            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)

            # 計算成本回彈次數
            if  i != 0 and self.cost_trigger != None:
                if self.eval_['cost'][-2] < cost:
                    self.cost_count += 1

            # 輸出訓練紀錄入至CLI上
            sys.stderr.write('\rEpoch:%0*d/%d | Cost: %.2f '
                             '| Train Acc.: %.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100))
            sys.stderr.flush()

            # 輸出訓練紀錄入至設定的UI上
            if self.out_Text_flag:
                if self.cost_trigger != None:
                    self.text_output_func('\nEpoch:\r%0*d/%d | Cost: %.2f '
                             '| Train Acc.: %.2f%% | Cost Trigger: %d' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100, self.cost_count))
                else:
                    self.text_output_func('\nEpoch:\r%0*d/%d | Cost: %.2f '
                             '| Train Acc.: %.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100))
            
            # 偵測是否已到達限制的辨識率
            if self.acc_limit != None: 
                if train_acc >= self.acc_limit:  # ACC. limit
                    return

            # 偵測是否已到達限制的成本下限
            if self.cost_limit != None: 
                if cost <= self.cost_limit:      # cost limit
                    return

            # early stop
            # 偵測是否已到達成本回彈限制數
            if  i != 0 and self.cost_trigger != None:
                if self.cost_count >= self.cost_trigger:
                    return

        return self

    def predict(self, X):
        """
        預測類別標籤

        參數
        -----------
        X : array, shape = [樣本數, 特徵數]
            輸入層與其原始標籤

        返回
        ----------
        y_pred : array, shape = [樣本數]
            預測類別標籤
        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
    
    def text_output_func(self, string):
        """
        輸出文字介面設定

        參數
        -----------
        string : string
            輸出字串
        """

        # 請自行改寫本處
        
        return 