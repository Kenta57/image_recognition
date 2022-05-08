import numpy as np
import layer
import mnist
import random
import optimizer

import matplotlib.pyplot as plt
from pylab import cm
from pathlib import Path

class Network:
    def __init__(self, layers, loss_layer, batch_size):
        self.network = layers
        self.loss_layer = loss_layer
        self.batch_size = batch_size

    def fit(self, input_shape, data_x, data_y, epoch, optimizer_name, save_flag=False, savedata_path ='sample'):
        self.__set_optimizer(optimizer_name)
        datasize = data_x.shape[0]
        for i in range(epoch*int(datasize/self.batch_size)):
            idxs = np.random.choice(range(datasize),self.batch_size)
            input = data_x[idxs]/255
            input = np.reshape(input, (self.batch_size,*input_shape))

            # predict
            x = self.__forward(input)

            # loss
            t = self.__one_hot(data_y, idxs)
            x = self.loss_layer.forward(x,t)

            self.network.reverse()

            # 誤差逆伝搬
            self.__backward()

            self.network.reverse()

            # 重みの修正
            self.__updata()

            if i%int(datasize/self.batch_size) == 0:
                print(f'---{int(i/(datasize/self.batch_size))+1}epoch---')
                print(f'loss:{x}')
                if save_flag:
                    self.save(savedata_path)

    def save(self, savedata_path):
        path = savedata_path
        path.mkdir(exist_ok=True)
        index = 0
        for layer in self.network:
            list_W = layer.getW()
            for w in list_W:
                np.save(str(path/f'W{index}'),w)
                index += 1

    def load(self, savedata_path):
        path = savedata_path
        path.mkdir(exist_ok=True)
        index = 0
        for layer in self.network:
            num = layer.numW()
            list_W = []
            for k in range(num):
                list_W.append(np.load(path/f'W{index+k}.npy'))
            layer.loadW(list_W)
            index += num

    def evaluate(self, input_shape, test_x, test_y):
        test_datasize = test_x.shape[0]
        correct_cnt = 0
        for i in range(test_datasize):
            img_array = test_x[i]
            input = img_array/255
            input = np.reshape(input,(1,*input_shape))

            # predict
            x = self.__forward(input,flag=True)
            ans = np.argmax(x)
            if ans == test_y[i]:
                correct_cnt += 1

        # for i in range(test_datasize//self.batch_size):
        #     img_array = test_x[i:i+self.batch_size]
        #     input = img_array/255
        #     input = np.reshape(input,(self.batch_size,*input_shape))
        #
        #     # predict
        #     x = self.__forward(input,flag=True)
        #     ans = np.argmax(x)
        #     if ans == test_y[i]:
        #         correct_cnt += 1

        accuracy = correct_cnt/test_datasize
        print(f'correct_rate:{accuracy}')

        return accuracy



    def __set_optimizer(self, optimizer_name):
        if optimizer_name == 'SGD':
            for layer in self.network:
                layer.set_optimizer(optimizer.SGD())
        elif optimizer_name == 'Momentum':
            for layer in self.network:
                layer.set_optimizer(optimizer.Mometum())
        elif optimizer_name == 'AdaGrad':
            for layer in self.network:
                layer.set_optimizer(optimizer.AdaGrad())
        elif optimizer_name == 'RMSProp':
            for layer in self.network:
                layer.set_optimizer(optimizer.RMSProp())
        elif optimizer_name == 'AdaDelta':
            for layer in self.network:
                layer.set_optimizer(optimizer.AdaDelta())
        elif optimizer_name == 'Adam':
            for layer in self.network:
                layer.set_optimizer(optimizer.Adam())



    def __forward(self, input, flag=False):
        for i,layer in enumerate(self.network):
            if i == 0:
                x = layer.forward(input, flag)
            else:
                x = layer.forward(x, flag)

        return x

    def __backward(self):
        dout = 1
        dout = self.loss_layer.backward(dout)
        for layer in self.network:
            dout = layer.backward(dout)

    def __updata(self):
        for layer in self.network:
            layer.update()

    def __one_hot(self, Y, data_idxs):
        return np.identity(10)[Y[data_idxs]]
