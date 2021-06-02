from collections import OrderedDict
from layers import *
from utils import *
from tqdm import tqdm

class MLP:
    def __init__(self, hidden1_size = 5, hidden2_size = 3, input_size = 2, output_size = 1):
        self.params = {'W1': np.random.normal(0, np.sqrt(2/(input_size+hidden1_size)), size=(input_size, hidden1_size)), 'b1': np.zeros(hidden1_size),
                       'W2': np.random.normal(0, np.sqrt(2/(hidden1_size+hidden2_size)), size=(hidden1_size, hidden2_size)), 'b2': np.zeros(hidden2_size),
                       'W3':np.random.normal(0, np.sqrt(2/(hidden2_size+output_size)), size=(hidden2_size, output_size)), 'b3': np.zeros(output_size)}

        self.layers = OrderedDict()
        self.layers['Hidden1'] = Hidden(self.params['W1'], self.params['b1'])
        self.layers['Sigmoid1'] = Sigmoid()
        self.layers['Hidden2'] = Hidden(self.params['W2'], self.params['b2'])
        self.layers['Sigmoid2'] = Sigmoid()
        self.layers['Hidden3'] = Hidden(self.params['W3'], self.params['b3'])
        self.layers['Sigmoid3'] = Sigmoid()
        self.lastLayer = Last()

        self.train_loss=[]

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.lastLayer.forward(y, t)
        return loss

    def gradient(self, x, t):
        self.loss(x, t)

        dout = self.lastLayer.backward()

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {'W1': self.layers['Hidden1'].dW, 'b1': self.layers['Hidden1'].db,
                 'W2': self.layers['Hidden2'].dW, 'b2': self.layers['Hidden2'].db,
                 'W3': self.layers['Hidden3'].dW, 'b3': self.layers['Hidden3'].db}

        return grads

    def fit(self, x_train, t_train, lr, epochs, batch_size=1):
        train_size = x_train.shape[0]
        iters = int(train_size / batch_size)

        for epoch in tqdm(range(epochs), desc='epochs'):
            for iter in range(iters):
                idx = np.random.choice(train_size, batch_size)
                x_batch = x_train[idx]
                t_batch = t_train[idx]

                grad = self.gradient(x_batch, t_batch)
                for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                    self.params[key] -= lr*grad[key]

                loss = self.loss(x_batch, t_batch)
                self.train_loss.append(loss)



def main():
    # data pre-paring
    train_data =get_data('Trn.txt')
    x_train, t_train = data_split(train_data)
    x_test = get_data('Tst.txt')

    # weight initialization
    network = MLP(100,50)

    # train parameter
    epoch = 500
    batch_size = 1 # fully-SGD
    lr = 1

    network.fit(x_train, t_train, lr, epoch, batch_size)

    # loss graph
    loss_plot(len(network.train_loss), network.train_loss, 'ReLu')

    # predict
    y_pred = network.predict(x_test)
    visualization(x_test, y_pred, x_train, t_train, 'ReLu')
    #visualization3D(x_test,y_pred)




if __name__ == '__main__':
    main()