import numpy as np


class SGD:
    def __init__(self, mu=0.01):
        self.mu = mu

    def update(self, param, grad):
        param = param - self.mu*grad
        return param

class Momentum:
    def __init__(self, eta=0.01, alpha=0.9):
        self.eta = eta
        self.alpha = alpha

        self.count = -1

        self.alphas = []
        self.deltaWs = []

    def update(self, param, grad, index):
        if self.count < index:
            self.count += 1
            self.alphas.append(self.alpha)
            self.deltaWs.append(np.zeros_like(param))

        self.deltaWs[index] = self.alphas[index]*self.deltaWs[index] - self.eta*grad
        param = param + self.deltaWs[index]
        return param

class AdaGrad:
    def __init__(self, eta=0.001, h=1e-8):
        self.eta = eta
        self.h = h

        self.count = -1

        self.etas = []
        self.hs = []

    def update(self, param, grad, index):
        if self.count < index:
            self.count += 1
            self.etas.append(self.eta)
            self.hs.append(self.h)

        self.hs[index] = self.hs[index] + grad*grad
        param = param - self.etas[index]*self.hs[index]**(-1/2)*grad
        return param

class RMSProp:
    def __init__(self, h=0, eta=0.001, rho=0.9, epsilon=1e-8):
        self.h = h
        self.eta = eta
        self.rho = rho
        self.epsilon = epsilon

        self.count = -1

        self.hs = []
        self.etas = []
        self.rhos = []
        self.epsilons = []

    def update(self, param, grad, index):
        if self.count < index:
            self.count += 1
            self.hs.append(self.h)
            self.etas.append(self.eta)
            self.rhos.append(self.rho)
            self.epsilons.append(self.epsilon)

        self.hs[index] = self.rhos[index]*self.hs[index] + (1-self.rhos[index])*grad*grad
        param = param - self.etas[index]/(self.hs[index]**(1/2)+self.epsilons[index])*grad
        return param

class AdaDelta:
    def __init__(self, rho=0.95, epsilon=1e-6):
        self.h = 0
        self.s = 0
        self.rho = rho
        self.epsilon = epsilon

        self.count = -1

        self.hs = []
        self.ss = []
        self.rhos = []
        self.epsilons = []

    def update(self, param, grad, index):
        if self.count < index:
            self.count += 1
            self.hs.append(self.h)
            self.ss.append(self.s)
            self.rhos.append(self.rho)
            self.epsilons.append(self.epsilon)

        self.hs[index] = self.rhos[index]*self.hs[index] + (1-self.rhos[index])*grad*grad
        deltaW = -(self.ss[index]+self.epsilons[index])**(1/2)/(self.hs[index]+self.epsilons[index])**(1/2)*grad
        self.ss[index] = self.rhos[index]*self.ss[index] + (1-self.rhos[index])*deltaW*deltaW
        param = param + deltaW
        return param

class Adam:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0

        self.count = -1

        self.alphas = []
        self.beta1s = []
        self.beta2s = []
        self.epsilons = []
        self.ts = []
        self.ms = []
        self.vs = []

    def update(self, param, grad, index):
        if self.count < index:
            self.count += 1
            self.alphas.append(self.alpha)
            self.beta1s.append(self.beta1)
            self.beta2s.append(self.beta2)
            self.epsilons.append(self.epsilon)
            self.ts.append(self.t)
            self.ms.append(self.m)
            self.vs.append(self.v)

        self.ts[index] = self.ts[index] + 1
        self.ms[index] = self.beta1s[index]*self.ms[index] + (1-self.beta1s[index])*grad
        self.vs[index] = self.beta2s[index]*self.vs[index] + (1-self.beta2s[index])*grad*grad
        m_conv = self.ms[index]/(1-self.beta1s[index]**self.ts[index])
        v_conv = self.vs[index]/(1-self.beta2s[index]**self.ts[index])
        param = param - self.alphas[index]*m_conv/(v_conv**(1/2)+self.epsilons[index])
        return param