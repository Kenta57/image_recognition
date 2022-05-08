import numpy as np
import layer
import mnist
import random
import optimizer
import network

from pathlib import Path
import pickle

# main
base_path = Path('/Users/murayamakenta/Documents/zikken4/save_data')
batch_size = 100
epoch = 100

# layer
layers = []

layers.append(layer.Convolution(32, 3, (3,32,32),padding=1))
layers.append(layer.ReLU())
layers.append(layer.MaxPooling(height=2,width=2))
layers.append(layer.Dropout(0.25))

layers.append(layer.Convolution(64, 3, (32,16,16),padding=1))
layers.append(layer.ReLU())
layers.append(layer.MaxPooling(height=2,width=2))
layers.append(layer.Dropout(0.25))

layers.append(layer.Affine(64*8*8,512))
layers.append(layer.ReLU())
layers.append(layer.Dropout(0.5))
layers.append(layer.Affine(512,10))

loss_layer = layer.SoftmaxWithLoss()

# model
model = network.Network(layers, loss_layer, batch_size=batch_size)

# data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='bytes')
    X = np.array(dict[b'data'])
    X = X.reshape((X.shape[0],3,32,32))
    Y = np.array(dict[b'labels'])
    return X,Y

X,Y = unpickle("/Users/murayamakenta/Documents/zikken4/cifar-10-batches-py/data_batch_1")
for i in range(2):
    x,y = unpickle(f'/Users/murayamakenta/Documents/zikken4/cifar-10-batches-py/data_batch_{i+2}')
    X = np.concatenate([X,x])
    Y = np.concatenate([Y,y])

# 学習
filename = 'cifar10_test'
savedata_path = base_path / filename
# loaddata
model.load(savedata_path=savedata_path)
# fit
model.fit(input_shape=(3,32,32),data_x=X, data_y=Y, epoch=epoch, optimizer_name='Adam', save_flag=True, savedata_path=base_path)

# 保存
model.save(savedata_path=savedata_path)

# 評価
test_X,test_Y = unpickle("/Users/murayamakenta/Documents/zikken4/cifar-10-batches-py/test_batch")
model.evaluate(input_shape=(3,32,32), test_x = test_X, test_y=test_Y)