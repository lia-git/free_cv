import mxnet.ndarray as nd
import mxnet.autograd as ad
import random
import mxnet as mx


ctx=mx.gpu(0)
try:
    nd.array([1,2,3], ctx=ctx)
except mx.MXNetError as err:
    ctx=mx.cpu(0)
    pass

def compute_loss(y_s, y):
    return (y_s - y.reshape(shape=(y_s.shape))) ** 2


def data_iter(X, Y, batch_size=50):
    length = len(X)
    idx = list(range(length))
    random.shuffle(idx)
    for i in range(0, length, batch_size):
        ix_ = nd.array(idx[i: min(length, i + batch_size)],ctx=ctx)
        yield nd.take(X, ix_), nd.take(Y, ix_)


def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad


def main():
    # generating dataset
    num_features = 5
    total = 1000
    weights = [1.5, -3.4, -2.6, 7.2, -3.0]
    biases = 2.6
    X = nd.random_normal(shape=(total, num_features),ctx=ctx)
    Y = weights[0] * X[:, 0] + weights[1] * X[:, 1] + weights[2] * X[:, 2] + weights[3] * X[:, 3] + weights[4] * X[:,
                                                                                                                 4] + biases
    # Y +=  nd.random_normal(shape=Y.shape)

    # iniitialize the parameters
    W_hat = nd.random_normal(shape=(num_features, 1),ctx=ctx)
    b_hat = nd.random_normal(shape=(1,),ctx=ctx)
    for i in [W_hat, b_hat]:
        i.attach_grad()

    # training
    epochs = 10
    lr = 0.001
    total_loss = 0
    for epoch in range(epochs):
        for x_, y_ in data_iter(X, Y):
            with ad.record():
                loss = compute_loss(nd.dot(x_ , W_hat) + b_hat, y_)
            loss.backward()

            SGD([W_hat, b_hat], lr)
            total_loss += nd.sum(loss).asscalar()
            # print("Epoch %d, average loss: %f" % (epoch, total_loss / total))
    print(W_hat, b_hat)

if __name__ == "__main__":
    main()
