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
        self.cost_count = 0

        if self.text_output == None:
            self.out_Text_flag = False
        else:
            self.out_Text_flag = True

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
            onehot[val, idx] = 1.
        return onehot.T

    def _sigmoid(self, z):
        """計算 logistic function (sigmoid)"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250))) # 限制 z 最多在 -250~250 之間
    
    def _forward(self, X):
        """計算 forward propagation 步驟"""

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

        cost = np.sum((y_enc - output)**2)/2.0
        return cost
    
    def fit(self, X_train, y_train):
        n_output = np.unique(y_train).shape[0] + 1 # number of class labels 
        # !!!!!
        # (because of the dataset, we need to add 1 class to represent 0 bit.)
        # if you think about that will influence the result, you can remove it, but you need check your dataset before you do.
        # !!!!!

        n_features = X_train.shape[1]

        if self.out_Text_flag:
            self.text_output_func('\n神經輸出: '+str(n_output-1)+'\n輸入特徵: '+str(n_features))
        #self.text_output.insert('insert','\n神經輸出: '+str(n_output)+'\n輸入特徵: '+str(n_features))

        ########################
        # Weight initialization
        ########################

        # input = [n_sample, n_features + 1] # 加 1 是減 bias(theta) 用

        # weights for input -> hidden
        self.b_h = np.zeros(self.n_hidden)
        self.w_h = self.random.normal(loc=0.0, scale=0.1,size=(n_features, self.n_hidden))

        # if self.out_Text_flag:
        #     self.text_output_func('\n第一層神經層鍵結值大小: '+str(self.w_h.shape))
        #     self.text_output_func('\n第一層神經層鍵結值: \n'+str(self.w_h))

        # weights for hidden -> output
        self.b_out = np.zeros(n_output)
        self.w_out = self.random.normal(loc=0.0, scale=0.1,size=(self.n_hidden, n_output))

        # if self.out_Text_flag:
        #     self.text_output_func('\n輸出層神經層鍵結值大小: '+str(self.w_out.shape))
        #     self.text_output_func('\n輸出層神經層鍵結值: \n'+str(self.w_out))

        epoch_strlen = len(str(self.epochs))  # for progress formatting
        self.eval_ = {'cost': [], 'train_acc': []}

        y_train_enc = self._onehot(y_train, n_output)

        # iterate over training epochs
        for i in range(self.epochs):

            # iterate over minibatches
            indices = np.arange(X_train.shape[0])

            if self.shuffle:
                self.random.shuffle(indices)

            for start_idx in range(0, indices.shape[0] - self.minibatch_size +
                                   1, self.minibatch_size):
                batch_idx = indices[start_idx:start_idx + self.minibatch_size]

                # forward propagation
                z_h, a_h, z_out, a_out = self._forward(X_train[batch_idx])

                ##################
                # Backpropagation
                ##################

                # [n_samples, n_classlabels]
                e_out = a_out - y_train_enc[batch_idx]

                # [n_samples, n_classlabels]
                sigmoid_derivative_o = a_out * (1. - a_out)

                # [n_samples, n_hidden]
                sigmoid_derivative_h = a_h * (1. - a_h)

                # [n_samples, n_classlabels] 
                detla_o = e_out * sigmoid_derivative_o

                # [n_samples, n_hidden] = [n_samples, n_hidden] * [n_samples, n_classlabels]  * [n_hidden, n_classlabels]
                delta_h = sigmoid_derivative_h * np.dot(detla_o, self.w_out.T)

                # [n_samples, n_hidden] = [n_samples, n_hidden] - eta * [n_samples, n_features].T dot [n_samples, n_hidden]
                self.w_h -= self.eta * np.dot(X_train[batch_idx].T, delta_h) 
                self.b_h -= self.eta * np.sum(delta_h, axis=0)
                
                # [n_hidden, n_classlabels] = [n_hidden, n_classlabels] - eta * [n_samples, n_hidden].T dot [n_samples, n_classlabels]  
                self.w_out -= self.eta * np.dot(a_h.T, detla_o)
                self.b_out -= self.eta * np.sum(e_out, axis=0)

            #############
            # Evaluation
            #############

            # Evaluation after each epoch during training

            z_h, a_h, z_out, a_out = self._forward(X_train)

            cost = self._compute_cost(y_enc=y_train_enc, output=a_out)
            #print(cost)

            y_train_pred = self.predict(X_train)

            train_acc = ((np.sum(y_train == y_train_pred)).astype(np.float64) /
                         X_train.shape[0])

            self.eval_['cost'].append(cost)
            self.eval_['train_acc'].append(train_acc)

            if  i != 0 and self.cost_trigger != None:
                if self.eval_['cost'][-2] < cost:
                    self.cost_count += 1

            sys.stderr.write('\rEpoch:%0*d/%d | Cost: %.2f '
                             '| Train Acc.: %.2f%% ' %
                             (epoch_strlen, i+1, self.epochs, cost,
                              train_acc*100))
            sys.stderr.flush()

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

            if self.acc_limit != None: 
                if train_acc >= self.acc_limit:  # ACC. limit
                    return
            if self.cost_limit != None: 
                if cost <= self.cost_limit:      # cost limit
                    return

            # early stop
            if  i != 0 and self.cost_trigger != None:
                if self.cost_count >= self.cost_trigger:
                    return

        return self

    def predict(self, X):
        """Predict class labels

        Parameters
        -----------
        X : array, shape = [n_samples, n_features]
            Input layer with original features.

        Returns:
        ----------
        y_pred : array, shape = [n_samples]
            Predicted class labels.

        """
        z_h, a_h, z_out, a_out = self._forward(X)
        y_pred = np.argmax(z_out, axis=1)
        return y_pred
    
    def text_output_func(self, string):
        # self.text_output.insert('insert',string)
        # self.text_output.see('end')
        return 