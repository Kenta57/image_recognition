import numpy as np

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x, flag=False):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx

    def set_optimizer(self, optimizer):
        pass

    def update(self):
        pass

    def getW(self):
        return []

    def numW(self):
        return 0

    def loadW(self, list_W):
        pass

class Affine:
    def __init__(self, input_node_num, output_node_num):
        self.W = self.__initial_weight(output_node_num,input_node_num)
        self.b = self.__initial_weight(output_node_num,input_node_num,bias=True)
        self.x = None
        self.dW = None
        self.db = None
        self.optimizer = None
        self.original_x_shape = None

    def forward(self, x, flag=False):
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)
        return dx

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update(self):
        self.W = self.optimizer.update(self.W, self.dW,0)
        self.b = self.optimizer.update(self.b, self.db,1)

    def getW(self):
        list_W = [self.W, self.b]
        return list_W

    def numW(self):
        return 2

    def loadW(self, list_W):
        self.W = list_W[0]
        self.b = list_W[1]

    def __initial_weight(self, curnode_num, prenode_num, bias=False):
        if bias != True:
            return np.random.normal(0.0,1/prenode_num,(prenode_num,curnode_num))
        elif bias:
            return np.random.normal(0.0,1/prenode_num,(1,curnode_num))

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = self.__softmax(x)
        loss = self.__cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx

    def __softmax(self,x):
        max_value = np.max(x, axis=-1, keepdims=True)
        x -= max_value
        return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

    def __cross_entropy_error(self,y,t):
        if y.ndim == 1:
            t = t.reshape(1,-1)
            y = y.reshape(1,-1)

        delta = 1e-7
        batch_size = y.shape[0]
        return -np.sum(t*np.log(y+delta))/batch_size


class ReLU:
    def __init__(self):
        self.out = None

    def forward(self, x, flag=False):
        self.out = np.where(x > 0, x, 0)
        return self.out

    def backward(self, dout):
        mask = np.where(self.out > 0, 1, 0)
        dx = dout*mask
        return dx

    def set_optimizer(self, optimizer):
        pass

    def update(self):
        pass

    def getW(self):
        return []

    def numW(self):
        return 0

    def loadW(self, list_W):
        pass

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, x, flag = False):
        if not flag:
            random = np.random.rand(*x.shape)
            self.mask = np.where(random > self.rate, 1, 0)
            out = self.mask
        else:
            out = x*(1-self.rate)
        return out

    def backward(self, dout):
        return dout*self.mask

    def set_optimizer(self, optimizer):
        pass

    def update(self):
        pass

    def getW(self):
        return []

    def numW(self):
        return 0

    def loadW(self, list_W):
        pass

class Batch_Normalization:
    def __init__(self, gamma=1, beta=0):
        self.gamma = gamma
        self.beta = beta
        self.epsilon = 1e-7
        self.exp_mean = None
        self.exp_var = None
        self.mean = None
        self.x = None
        self.x_conv = None
        self.var = None
        self.dgamma = None
        self.dbeta = None

        self.optimizer = None

    def forward(self, x, flag = False):
        self.x = x
        if self.exp_mean is None:
            self.exp_mean = np.zeros(x.shape[1])
            self.exp_var = np.zeros(x.shape[1])

        if not flag:
            batch_size = x.shape[0]
            mean = np.sum(x,axis=0)/batch_size
            var = np.sum((x - mean)**2,axis=0)/batch_size
            x_conv = (x - mean)/np.sqrt(var+self.epsilon)
            y = self.gamma*x_conv + self.beta
            # 期待値の近似
            self.exp_mean = 0.9*self.exp_mean + 0.1*mean
            self.exp_var = 0.9*self.exp_var + 0.1*var

            self.mean = mean
            self.var = var
            self.x_conv = x_conv
            return y
        else:
            x_conv = (x - self.exp_mean)/np.sqrt(self.exp_var + self.epsilon)
            y = self.gamma*x_conv + self.beta
            return y


    def backward(self, dout):
        batch_size = self.x.shape[0]
        dx_conv = dout*self.gamma
        dsigma = np.sum(-1/2*dx_conv*(self.x-self.mean)*(self.var + self.epsilon)**(-3/2),axis=0)
        dmean = np.sum(-dx_conv*(self.var + self.epsilon)**(-1/2),axis=0) + dsigma*np.sum(-2*(self.x - self.mean),axis=0)/batch_size
        dx = dx_conv*(self.var + self.epsilon)**(-1/2) + dsigma*2*(self.x - self.mean)/batch_size + dmean/batch_size
        dgamma = np.sum(dout*self.x_conv,axis=0)
        dbeta = np.sum(dout,axis=0)

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update(self):
        self.gamma = self.optimizer.update(self.gamma, self.dgamma,0)
        self.beta = self.optimizer.update(self.beta, self.dbeta,1)

    def getW(self):
        list_W = [self.gamma, self.beta, self.exp_mean, self.exp_var]
        return list_W

    def numW(self):
        return 4

    def loadW(self, list_W):
        self.gamma = list_W[0]
        self.beta = list_W[1]
        self.exp_mean = list_W[2]
        self.exp_var = list_W[3]

def img2col(x, filter_h, filter_w, padding=0, stride=1):
    batch, channel, height, width = x.shape
    OH = (height + 2*padding - filter_h) // stride + 1
    OW = (width + 2*padding - filter_w) // stride + 1


    # x_pad = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    # col = np.zeros((batch, OH*OW, filter_h*filter_w*channel))
    #
    # for batch_i in range(batch):
    #     height_index = 0
    #     pos_h = -stride
    #     for k in range(OH):
    #         pos_h += stride
    #         pos_w = -stride
    #         for l in range(OW):
    #             pos_w += stride
    #             pre_tensor = x_pad[batch_i,: ,pos_h:pos_h+filter_h,pos_w:pos_w+filter_w]
    #             pre_col = np.reshape(pre_tensor,(1,-1))
    #             col[batch_i,height_index,:] = pre_col
    #             height_index += 1

    x_pad = np.pad(x,((0,0),(0,0),(padding,padding),(padding,padding)))
    col = np.zeros((batch, OH*OW, filter_h*filter_w*channel))

    height_index = 0
    pos_h = -stride
    for k in range(OH):
        pos_h += stride
        pos_w = -stride
        for l in range(OW):
            pos_w += stride
            pre_tensor = x_pad[:,: ,pos_h:pos_h+filter_h,pos_w:pos_w+filter_w]
            pre_col = np.reshape(pre_tensor,(batch,-1))
            col[:,height_index,:] = pre_col
            height_index += 1

    col = np.reshape(col, (batch*OH*OW,-1))
    return col

def col2img(x, base_shape, filter_h, filter_w, padding=0, stride=1):
    batch, channel, height, width = base_shape
    OH = (height + 2*padding - filter_h) // stride + 1
    OW = (width + 2*padding - filter_w) // stride + 1

    img = np.zeros((batch, channel, height + 2*padding + stride -1, width + 2*padding + stride - 1))

    # for batch_i in range(batch):
    #     height_index = 0
    #     pos_h = -stride
    #     for k in range(OH):
    #         pos_h += stride
    #         pos_w = -stride
    #         for l in range(OW):
    #             pos_w += stride
    #             pre_tensor = np.reshape(x[height_index,:],(channel,filter_h,filter_w))
    #             img[batch_i,:,pos_h:pos_h+filter_h,pos_w:pos_w+filter_w] += pre_tensor
    #             height_index += 1

    height_index = 0
    pos_h = -stride
    x = np.reshape(x,(batch,OH*OW,-1))
    for k in range(OH):
        pos_h += stride
        pos_w = -stride
        for l in range(OW):
            pos_w += stride
            pre_tensor = np.reshape(x[:,height_index,:],(batch, channel, filter_h, filter_w))
            img[:,:,pos_h:pos_h+filter_h,pos_w:pos_w+filter_w] += pre_tensor
            height_index += 1

    return img[:,:,padding:padding+height,padding:padding+width]

class Convolution:
    def __init__(self, num_filter, size_filter, input_shape, padding=0, stride=1):
        self.W = self.__initial_weight(num_filter, input_shape[0], size_filter)
        self.b = self.__initial_weight(num_filter, input_shape[0], size_filter, bias = True)
        self.padding = padding
        self.stride = stride

        self.x = None
        self.col = None
        self.col_filter = None

        self.dW = None
        self.db = None

    def forward(self, x, flag = False):
        num_f, channel_f, height_f, width_f = self.W.shape
        batch, channel, height, width = x.shape
        OH = (height + 2*self.padding - height_f) // self.stride + 1
        OW = (width + 2*self.padding - width_f) // self.stride + 1

        col = img2col(x, height_f, width_f, self.padding, self.stride)
        col_filter = self.W.reshape(num_f, -1).T

        out = np.dot(col,col_filter) + self.b
        out = out.T
        out = np.reshape(out, (num_f, batch, OH, OW)).transpose(1,0,2,3)

        self.x = x
        self.col = col
        self.col_filter = col_filter

        return out

    def backward(self, dout):
        num_f, channel_f, height_f, width_f = self.W.shape
        dout = np.reshape(dout.transpose(0,2,3,1),(-1, num_f))

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = np.reshape(self.dW.T,(num_f, channel_f, height_f, width_f))

        dx = np.dot(dout, self.col_filter.T)
        dx = col2img(dx, self.x.shape, height_f, width_f, self.padding, self.stride)

        return dx

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def update(self):
        self.W = self.optimizer.update(self.W, self.dW,0)
        self.b = self.optimizer.update(self.b, self.db,1)

    def getW(self):
        list_W = [self.W, self.b]
        return list_W

    def numW(self):
        return 2

    def loadW(self, list_W):
        self.W = list_W[0]
        self.b = list_W[1]

    def __initial_weight(self, num_filter, channel_f, size_filter, bias=False):
        if bias != True:
            return np.random.normal(0.0, 0.01, (num_filter, channel_f, size_filter, size_filter))
        elif bias:
            return np.zeros(num_filter)


class MaxPooling:
    def __init__(self, height, width, padding=0, stride=2):
        self.height = height
        self.width = width
        self.padding = padding
        self.stride = stride

        self.x = None
        self.arg_max = None

    def forward(self, x, flag=False):
        batch, channel, height, width = x.shape
        out_h = round((height - self.height + self.stride -1) / self.stride)
        out_w = round((width - self.width + self.stride -1) / self.stride)

        col = img2col(x, self.height, self.width, self.padding, self.stride)
        col = np.reshape(col, (-1,self.height*self.width))

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(batch, out_h, out_w, channel).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = np.reshape(dout, (-1,1))
        pool_size = self.height * self.width
        col = np.zeros((dout.size,pool_size))
        a = np.reshape(np.arange(dout.size),(-1,1))
        b = np.reshape(self.arg_max,(-1,1))
        col[a,b] = dout
        batch, channel, height, width = self.x.shape
        out_h = round((height - self.height + self.stride -1) / self.stride)
        out_w = round((width - self.width + self.stride -1) / self.stride)
        col = np.reshape(col, (batch, channel, out_h*out_w, -1)).transpose(0,2,1,3)
        col = np.reshape(col, (batch*out_h*out_w, -1))

        dx = col2img(col, self.x.shape, self.height, self.width, self.padding, self.stride)

        return dx

    def set_optimizer(self, optimizer):
        pass

    def update(self):
        pass

    def getW(self):
        return []

    def numW(self):
        return 0

    def loadW(self, list_W):
        pass