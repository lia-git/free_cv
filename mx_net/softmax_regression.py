from mxnet import gluon
from mxnet import ndarray as nd


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
import matplotlib.pyplot as plt


class Net:
    def __init__(self, input_dim, output_size):
        self.weights = nd.random_normal(shape=(input_dim, output_size))
        self.bias = nd.zeros(shape=(1, output_size))
        self.params = [self.weights, self.bias]
        self.pre_act = None
        self.pro_act = None

    def soft_max(self, data):
        self.pre_act = nd.dot(data, self.weights) + self.bias
        tmp = nd.exp(self.pre_act)
        sum_tmp = tmp.sum(axis=1)
        self.pro_act = tmp / sum_tmp
        
    def cross_entropy(self, label):
        return - nd.pick(nd.log(self.pro_act), y)


def show_image(data):
    n = data.shape[0]
    _, figs = plt.subplots(1, n, figsize=[15, 15])
    for i in range(n):
        pixels = data[i].reshape((28, 28)).asnumpy()
        figs[i].imshow(pixels)
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()


def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]


def main():
    data, label = mnist_train[0:9]
    show_image(data)
    print(get_text_labels(label))
    batch_size = 256
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)
    
    epochs = 5
    learning_rate = .02
    net = 
    for epoch in range(epochs):
        for input,label in train_data:
            


if __name__ == '__main__':
    main()
