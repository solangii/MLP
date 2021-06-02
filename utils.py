import matplotlib.pyplot as plt
import numpy as np

def get_data(filename):
    file = open(filename, 'r').read().split('\n')
    dataset = []
    for i in range(len(file)):
        data = file[i].split(' ')
        if len(data) == 1:
            break

        for j in range(len(data)):
            data[j] = float(data[j])
        dataset.append(data)
    dataset = np.array(dataset)
    return dataset

def data_split(data):
    x = []
    y = []
    for i in range(data.shape[0]):
        x.append(data[i][0:2])
        y.append(data[i][2])

    x = np.array(x)
    y = np.array(y)
    return x, y

def visualization(x_test, y_pred, x_train, t_train, name):
    x0 = x_train[t_train == 0]
    x1 = x_train[t_train == 1]

    x = x_test[:,0]
    y = x_test[:,1]

    plt.xlim(-15.0, 15.0)
    plt.ylim(-15.0, 15.0)

    plt.plot(x0[:, 0], x0[:, 1], 'r')
    plt.plot(x1[:, 0], x1[:, 1], 'b')

    plt.scatter(x, y, c=y_pred, cmap='rainbow')
    plt.colorbar()
    plt.savefig('img/'+name+'.jpg', dpi=300)
    plt.show()


def visualization3D(x_test, y_pred):
    x = x_test[:, 0]
    y = x_test[:, 1]

    ax = plt.subplot(projection='3d')
    ax.scatter(x, y, y_pred, marker='o', alpha=0.1, cmap ='Greens')
    plt.savefig('img/3d_plot.jpg', dpi=300)
    plt.show()


def loss_plot(x, y, name):
    x = np.arange(x)
    plt.plot(x, y)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.savefig('img/'+name+'.jpg', dpi=300)
    plt.show()

def train_plot(x_train, t_train):
    x0 = x_train[t_train == 0]
    x1 = x_train[t_train == 1]

    plt.xlim(-15.0, 15.0)
    plt.ylim(-15.0, 15.0)
    plt.plot(x0[:,0], x0[:,1], 'r')
    plt.plot(x1[:,0], x1[:,1], 'b')
    plt.show()
