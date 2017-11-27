import mxnet.ndarray as nd
import mxnet.autograd as ad
import random


def compute_loss(y_s,y):
  return (y_s - y.reshape(shape=(y_s.shape)))**2

def data_iter(X, Y, batch_size = 50)
  len = len(X)
  idx = list(range(len))
  random.shuffle(idx)
  for i in range(0, len, batch_size):
      ix_ = idx[i : min(len, i + batch_size)]
      yield nd.take(X,idx_), nd.take(Y, idx_)

def SGD(params,lr):
  for param in params:
    param = param - lr * param.grad
  
  
      
  
def main():
    # generating dataset
    num_featrues = 5
    total = 1000
    weights = [1.5,3.4,2.6,7.2,3.0]
    biases = [1.23]
    X = nd.random_normal(shape = (total,num_featrues))
    Y = weights[0] * X[:,0] + weights[1] * X[:,1] + weights[2] * X[:,2] + weights[3] * X[:,3] +weights[4] * X[:,4] + biases
    Y += .001 * nd.random_normal(shape = Y.shape)
    
    
    # iniitialize the parameters
    W_hat = nd.random_normal(shape = (num_features, 1))
    b_hat = nd.random_normal(shape = (1, ))
    fot i in [W_hat, b_hat]:
      i.attach_grad()
    
    
    #training
    epochs = 5
    lr = 0.2
    total = 0 
    for epoch in range(epochs):
      for x_,y_ in data_iter(X,Y):
        with ad.record():
          loss = compute_loss(x_ * w_hat + b_hat,y_hat)
        loss.backward()
        
    
        SGD(params, learning_rate)
    total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_examples))
    
    
    

if __name__=="__main__":
    main()
