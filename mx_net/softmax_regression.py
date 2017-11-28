from mxnet import gluon
from mxnet import ndarray as nd


def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')


mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)
import matplotlib.pyplot as plt

def show_image(data):
    n = data.shape[0]
    _, figs = plt.subplots(1, n, figsize=[15, 15])
    for i in range(n):
        figs[i].imshow(data[i].reshape((28,28)))
        figs[i].axes.get_xaxis().set_visible(false)
        figs[i].axes.get_yaxis().set_visible(false)
    plt.show()

        

def main():
    def get_text_labels(label):
    text_labels = [
        't-shirt', 'trouser', 'pullover', 'dress,', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    ]
    return [text_labels[int(i)] for i in label]

    data, label = mnist_train[0:9]
    show_images(data)
    print(get_text_labels(label))
    batch_size = 256
    train_data = gluon.data.DataLoader(mnist_train, batch_size, shuffle=True)
    test_data = gluon.data.DataLoader(mnist_test, batch_size, shuffle=False)




if __name__ == '__main__':
    main()
