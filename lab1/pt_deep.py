import torch, data_l1, pt_logreg
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

class BatchNormLayer(nn.Module):
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        super(BatchNormLayer, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, num_features))  # Scale
        self.beta = nn.Parameter(torch.zeros(1, num_features))  # Shift
        self.momentum = momentum
        self.eps = eps
        
        # Running mean and variance
        self.running_mean = torch.zeros(1, num_features)
        self.running_var = torch.ones(1, num_features)
        
    def forward(self, X, training=True):
        if training:
            batch_mean = X.mean(dim=0, keepdim=True)
            batch_var = X.var(dim=0, unbiased=False, keepdim=True)
            
            # Normalization of the output
            X_norm = (X - batch_mean) / torch.sqrt(batch_var + self.eps)
            
            # Updating mean and variance based on mini-batch
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
        else:
            # Using existing mean and variance for evaluation normalization
            X_norm = (X - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        return self.gamma * X_norm + self.beta


class PTDeep(nn.Module):
  def __init__(self, n_neurons, act_func='relu', bn=True):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    super(PTDeep, self).__init__()
    self.bn = bn
    self.biases = nn.ParameterList()
    self.weights = nn.ParameterList()
    self.batch_norms = nn.ModuleList()
  	
    if (act_func == 'relu'):
      self.fn = torch.relu
    elif(act_func == 'sigmoid'):
      self.fn = torch.sigmoid
    elif(act_func == 'tanh'):
      self.fn = torch.tanh
    else:
      raise ValueError('Invalid activation function.')
  
    for i in range(len(n_neurons) - 1):
      self.weights.append(nn.Parameter(torch.randn(n_neurons[i], n_neurons[i+1]) * 0.1))
      self.biases.append(nn.Parameter(torch.zeros(1, n_neurons[i+1])))
      if bn and i < len(n_neurons) - 2:
                self.batch_norms.append(BatchNormLayer(n_neurons[i + 1]))
        
  def forward(self, X, training=True):
    h = X

    if self.bn:
      for i in range(len(self.weights) - 1):
        h = torch.mm(h, self.weights[i]) + self.biases[i]
        h = self.batch_norms[i].forward(h, training)
        h = self.fn(h)
    else:
       for i in range(len(self.weights) - 1):
        h = torch.mm(h, self.weights[i]) + self.biases[i]
        h = self.fn(h)

    scores = torch.mm(h, self.weights[-1]) + self.biases[-1]
    return torch.softmax(scores, dim=1)

  def get_loss(self, X, Yoh_, lbd=None, training=True):
    probs = self.forward(X, training)
    log_probs = torch.log(probs)
    reg_loss = 0
    if(lbd is not None):  
      reg_loss = lbd * sum(torch.sum(w ** 2) for w in self.weights)
    loss = -torch.mean(torch.sum(Yoh_ * log_probs, dim=1))
    return loss + reg_loss
  
def train(model, X, Yoh_, param_niter, param_delta, param_lambda=None):
  optimizer = optim.SGD(model.parameters(), lr=param_delta)
    
  for i in range(int(param_niter)):
    optimizer.zero_grad()
    loss = model.get_loss(X, Yoh_, lbd = param_lambda)
    loss.backward()
    optimizer.step()
        
    if i % 10 == 0:
      print(f'Step: {i}, Loss: {loss.item()}')

def pt_deep_decfun(model):
    def classify(X):
        probs = eval(model,X)
        if C == 2: # Binary probs for gradient mesh **useProbs=True**
            return probs[:, 1]
        return np.argmax(probs, axis=1) # Class preds for single color mesh **useProbs=False**
    return classify

def count_params(model):
    total_params = 0
    print("----------\nParameters:")
    for name, param in model.named_parameters():
      param_count = param.numel()
      total_params += param_count
      print(f"{name}: {list(param.shape)} ({param_count} parameters)")
    print(f"Total: {total_params}\n----------")

def eval(model, X, training=False):
  X_tensor = torch.tensor(X, dtype=torch.float32)
  probs = model.forward(X_tensor, training)
  return probs.detach().numpy()

if __name__ == "__main__":
  np.random.seed(100)

  K = 6
  C = 2
  N = 50
  X, Y_ = data_l1.sample_gmm_2d(K, C, N)

  X_tensor = torch.tensor(X, dtype=torch.float32) 
  Y_tensor = torch.tensor(Y_, dtype=torch.long)
  Yoh_ = F.one_hot(Y_tensor, num_classes=C).float()

  D = 2
  arr = []
  H = [10, 10] # Hidden layers
  arr.append(D)
  for h in H:
    arr.append(h)
  arr.append(C)

  ptlr = PTDeep(arr, 'relu', bn=True) # Model init, use bn=True for Batch Normalization

  train(ptlr, X_tensor, Yoh_, param_niter=1e3, param_delta=0.1, param_lambda=1e-4)

  probs = eval(ptlr, X)
  Y = np.argmax(probs, axis = 1)

  accuracy, cm, recall, precision = pt_logreg.eval_perf_multi(Y, Y_)
  print(f"\nAccuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nConfusion matrix:\n {cm}\n")
  print(count_params(ptlr))

  decfun = pt_deep_decfun(ptlr)
  bbox=(np.min(X, axis=0), np.max(X, axis=0))

  useProbs = False
  if(C == 2):
    useProbs = True
    
  data_l1.graph_surface(decfun, bbox, offset=0.5, useProbs=useProbs)
  data_l1.graph_data(X, Y_, Y)
  plt.show()