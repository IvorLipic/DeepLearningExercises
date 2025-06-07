import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import data_l1

class PTLogreg(nn.Module):
  def __init__(self, D, C):
    """Arguments:
       - D: dimensions of each datapoint 
       - C: number of classes
    """
    super(PTLogreg, self).__init__()
    # inicijalizirati parametre (koristite nn.Parameter):
    # imena mogu biti self.W, self.b
    self.b = nn.Parameter(torch.zeros(1, C)) # (1 x C)
    self.W = nn.Parameter(torch.randn(D, C)) # (D x C)

  def forward(self, X):
    # unaprijedni prolaz modela: izračunati vjerojatnosti
    #   koristiti: torch.mm, torch.softmax
    scores = torch.mm(X, self.W) + self.b # (N x D) x (D x C) -> (N x C)
    return torch.softmax(scores, dim=1)

  def get_loss(self, X, Yoh_, lbd=None):
    # formulacija gubitka
    #   koristiti: torch.log, torch.exp, torch.sum
    #   pripaziti na numerički preljev i podljev
    probs = self.forward(X)
    log_probs = torch.log(probs)
    if (lbd != None): 
      return torch.mean(-torch.sum(Yoh_ * log_probs, dim=1)) + lbd/2 * torch.norm(self.W)
    return torch.mean(-torch.sum(Yoh_ * log_probs, dim=1))


def train(model, X, Yoh_, param_niter, param_delta, param_lambda):
  """Arguments:
     - X: model inputs [NxD], type: torch.Tensor
     - Yoh_: ground truth [NxC], type: torch.Tensor
     - param_niter: number of training iterations
     - param_delta: learning rate
  """
  # inicijalizacija optimizatora
  optimizer = optim.SGD([model.b, model.W], lr=param_delta)

  # petlja učenja
  # ispisujte gubitak tijekom učenja
  for i in range(param_niter):
    
    optimizer.zero_grad()
    loss = model.get_loss(X, Yoh_, lbd=param_lambda)
    loss.backward()

    if i % 10 == 0:
        print(f'step: {i}, loss: {loss.item()}')
        
    optimizer.step()



def eval(model, X):
  """Arguments:
     - model: type: PTLogreg
     - X: actual datapoints [NxD], type: np.array
     Returns: predicted class probabilites [NxC], type: np.array
  """
  # ulaz je potrebno pretvoriti u torch.Tensor
  # izlaze je potrebno pretvoriti u numpy.array
  # koristite torch.Tensor.detach() i torch.Tensor.numpy()
  X_tensor = torch.tensor(X, dtype=torch.float32)
  probs = model.forward(X_tensor)
  return probs.detach().numpy()

def eval_perf_multi(Y, Y_):
    
    C = int(max(np.max(Y), np.max(Y_)) + 1)
    
    confusion_matrix = np.zeros((C, C), dtype=int)
    
    for pred, true in zip(Y, Y_):
        confusion_matrix[true, pred] += 1
    
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    
    precision = np.zeros(C)
    recall = np.zeros(C)
    
    for i in range(C):
        sum_row = np.sum(confusion_matrix[i, :])
        sum_col = np.sum(confusion_matrix[:, i])
        
        recall[i] = confusion_matrix[i, i] / sum_row if sum_row != 0 else 0.0
        precision[i] = confusion_matrix[i, i] / sum_col if sum_col != 0 else 0.0
    
    return accuracy, confusion_matrix, precision, recall

def pt_logreg_decfun(model):
    def classify(X):
        probs = eval(model,X)
        if C == 2: # Binary probs for gradient mesh **useProbs=True**
            return probs[:, 1]
        return np.argmax(probs, axis=1) # Class preds for single color mesh **useProbs=False**
    return classify


if __name__ == "__main__":
  #np.random.seed(100)

  K = 5
  C = 3
  N = 40
  X, Y_ = data_l1.sample_gmm_2d(K, C, N)

  X_tensor = torch.tensor(X, dtype=torch.float32) 
  Y_tensor = torch.tensor(Y_, dtype=torch.long)
  Yoh_ = F.one_hot(Y_tensor, num_classes=C).float()

  ptlr = PTLogreg(X.shape[1], C)

  train(ptlr, X_tensor, Yoh_, param_niter=1000, param_delta=0.01, param_lambda=0.1)

  probs = eval(ptlr, X)
  Y = np.argmax(probs, axis = 1)

  accuracy, cm, recall, precision = eval_perf_multi(Y, Y_)
  print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nConfusion matrix:\n {cm}")

  decfun = pt_logreg_decfun(ptlr)
  bbox=(np.min(X, axis=0), np.max(X, axis=0))

  useProbs = False
  if(C == 2):
    useProbs = True
    
  data_l1.graph_surface(decfun, bbox, offset=0.5, useProbs=useProbs)
  data_l1.graph_data(X, Y_, Y)
  plt.show()