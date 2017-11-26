import mxnet.ndarray as nd
import mxnet.autograd as ad
import random
def data_iter(X, Y, batch_size = 50)
  len = len(X)
  idx = list(range(len))
  random.shuffle(idx)
  for i in range(0, len, batch_size):
      ix_ = idx[i : min(len, i + batch_size)]
      yield nd.take(X,idx_), nd.take(Y, idx_)

def main():
    num_featrues = 5
    total = 1000
    weights = [1.5,3.4,2.6,7.2,3.0]
    biases = [1.23]
    X = nd.random_normal(shape=(total,num_featrues))
    Y = weights[0] * X[:,0] + weights[1] * X[:,1] + weights[2] * X[:,2] + weights[3] * X[:,3] +weights[4] * X[:,4] + biases
    Y += .001 * nd.random_normal(shape=Y.shape)
    
    
    

if __name__=="__main__":
    main()
