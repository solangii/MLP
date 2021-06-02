import numpy as np

class Hidden:
    def __init__(self, w, b):
        self.W = w
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self,x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1.0/(1.0+np.exp(-x))
        return self.y

    def backward(self, dout):
        return dout * self.y * (1.0 - self.y)

class ReLU:
    def __init__(self):
        self.x = None

    def forward(self, x):
        self.x = (x>0).astype(np.int)
        return x * self.x

    def backward(self, dout):
        return dout * self.x

class Last:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = self.identity(x)
        self.loss = self.mse(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        #batch_size = self.t.shape[0]
        #dx = 2*(self.y - self.t) / batch_size
        dx = 2 * (self.y - self.t)
        return dx

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def mse(y, t):
        result = np.square(np.subtract(y, t)).mean()
        return result
