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
# ネットワーク構築
input_node_num = 28*28
middle_node_num = 100
output_node_num = 10

# layer
layers = []

layers.append(layer.Affine(input_node_num, middle_node_num))
layers.append(layer.Sigmoid())
layers.append(layer.Affine(middle_node_num, output_node_num))

loss_layer = layer.SoftmaxWithLoss()

# model
model = network.Network(layers, loss_layer, batch_size=batch_size)

# data
X = mnist.download_and_parse_mnist_file("train-images-idx3-ubyte.gz")
Y = mnist.download_and_parse_mnist_file("train-labels-idx1-ubyte.gz")

# 学習
filename = 'main'
savedata_path = base_path / filename
# loaddata
model.load(savedata_path=savedata_path)
# fit
# model.fit(input_shape=(1,28,28),data_x=X, data_y=Y, epoch=epoch, optimizer_name='Adam', save_flag=True, savedata_path=savedata_path)

# 保存
# model.save(savedata_path=savedata_path)

# 評価
test_X = mnist.download_and_parse_mnist_file("t10k-images-idx3-ubyte.gz")
test_Y = mnist.download_and_parse_mnist_file("t10k-labels-idx1-ubyte.gz")
model.evaluate(input_shape=(1,28,28), test_x = test_X, test_y=test_Y)