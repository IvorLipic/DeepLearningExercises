import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import data_l1

def relu(x):
   return np.maximum(0, x)

def fcann2_train(X, Y, param_niter, param_delta, H=3):
    # X -> (N x D)
    Y = Y.astype(int)
    C = np.max(Y) + 1
    N = X.shape[0]
    D = X.shape[1]
    b1 = np.zeros((1, H)) # (1 x H)
    b2 = np.zeros((1, C)) # (1 x C)
    W1 = np.random.randn(D, H) * np.sqrt(2./ D) # (D x H)
    W2 = np.random.randn(H, C) * np.sqrt(2./ H)  # (H x C)

    ##
    #Iverson
    ##
    I_Y = np.zeros((N, C)) 
    I_Y[np.arange(N), Y] = 1
    #I_Y = np.eye(C)[Y] # <-- Same result

    # Propagate fw+bw
    for i in range(int(param_niter)):
      
      s1 = np.dot(X, W1) + b1 # (N x D) x (D x H) --> (N x H) + b1(1 x H)
      h1 = relu(s1) # (N x H)
      s2 = np.dot(h1, W2) + b2  # (N x H) x (H x C) --> (N x C) + b2(1 x C)
      s2 -= np.max(s2, axis = 1, keepdims=True) # (N x C)
      expscores = np.exp(s2) # (N x C)

      # nazivnik sofmaksa
      sumexp = np.sum(expscores, axis = 1, keepdims=True) # (N x 1)

      # logaritmirane vjerojatnosti razreda 
      probs = expscores / sumexp # (N x C)
      logprobs = np.log(probs) # (N x C)

      # gubitak
      loss = -np.sum(I_Y * logprobs) / N # scalar

      # dijagnostički ispis
      if i % 50 == 0:
          accuracy = np.mean(np.argmax(probs, axis=1) == Y)
          print(f"Iteration {i}: Loss = {loss:.4f}, Accuracy = {accuracy:.4%}")
      
      # 1. gradijenti gubitka linearne mjere drugog sloja
      grad_s2 = probs - I_Y # (N x C)
      
      # 2. gradijenti parametara drugog sloja
      grad_W2 = np.dot(h1.T, grad_s2) # (H x N) x (N x C) --> (H x C)
      grad_b2 = np.sum(grad_s2, axis = 0, keepdims=True) # (1 x C)

      # 3. gradijenti nelinearnog izlaza prvog sloja
      grad_h1 = np.dot(grad_s2, W2.T) # (N x H)
      
      # 4. gradijenti linearne mjere prvog sloja
      grad_s1 = grad_h1 * (s1 > 0) # (N x H)

      # 5. gradijenti parametara prvog sloja
      grad_W1 = np.dot(X.T, grad_s1) # (D x N) x (N x H) -> (D x H)
      grad_b1 = np.sum(grad_s1, axis = 0, keepdims=True) # (1 x H)

      # dijagnostički ispis
      if i % 50 == 0:
          print("grad_W1 norm:", np.linalg.norm(grad_W1), "|#|", "grad_W2 norm:", np.linalg.norm(grad_W2), "\n\n")

      # poboljšani parametri
      W1 += -param_delta * grad_W1 # (D x H) + (D x H)
      b1 += -param_delta * grad_b1 # (1 x H) + (1 x H)
      W2 += -param_delta * grad_W2 # (H x C) + (H x C)
      b2 += -param_delta * grad_b2 # (1 x C) + (1 x C)

    return W1, W2, b1, b2

def fcann2_classify(X, W1, W2, b1, b2):
    s1 = np.dot(X, W1) + b1 # (N x D) x (D x H) --> (N x H) + b1(1 x H)
    h1 = relu(s1) # (N x H)
    s2 = np.dot(h1, W2) + b2  # (N x H) x (H x C) --> (N x C) + b2(1 x C)
    s2 -= np.max(s2, axis = 1, keepdims=True) # (N x C)
    expscores = np.exp(s2) # (N x C)
    sumexp = np.sum(expscores, axis = 1, keepdims=True) # (N x 1)
    probs = expscores / sumexp # (N x C)
    return probs

def fcann2_decfun(W1, W2, b1, b2, C):
    def classify(X):
        probs = fcann2_classify(X, W1, W2, b1, b2)

        if C == 2: # Binary probs for gradient mesh **useProbs=True**
            return probs[:, 1]
        
        return np.argmax(probs, axis=1) # Class preds for single color mesh **useProbs=False**
    return classify

if __name__=="__main__":
    #np.random.seed(100)

    GCS = 5
    C = 2
    N = 10

    X,Y_ = data_l1.sample_gmm_2d(GCS, C, N)
    Y_ = Y_.ravel()

    W1, W2, b1, b2 = fcann2_train(X, Y_, param_niter=1e3, param_delta=0.001, H = 20)

    probs = fcann2_classify(X, W1, W2, b1, b2) # (N x C)
    Y = np.argmax(probs, axis = 1) # (N x 1)

    decfun = fcann2_decfun(W1, W2, b1, b2, C)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))

    useProbs = False
    if(C == 2):
       useProbs = True
    
    data_l1.graph_surface(decfun, bbox, offset=0.5, useProbs=useProbs)
    data_l1.graph_data(X, Y_, Y)
    plt.show()