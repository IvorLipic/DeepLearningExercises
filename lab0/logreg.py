import numpy as np
import data
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def logreg_train(X, Y, param_niter, param_delta):
    # eksponencirane klasifikacijske mjere
    # pri računanju softmaksa obratite pažnju
    # na odjeljak 4.1 udžbenika
    # (Deep Learning, Goodfellow et al)!
    Y = Y.astype(int)
    C = np.max(Y) + 1
    N = X.shape[0]
    D = X.shape[1]
    b = np.zeros((C, 1)) # C x 1
    W = np.random.randn(C, D) # C x D

    I_Y = np.zeros((N, C))
    I_Y[np.arange(N), Y] = 1

    W_history = []
    b_history = []

    for i in range(param_niter):

      scores = np.dot(X, W.T) + b.T # (N x C)
      scores -= np.max(scores, axis = 1, keepdims=True)  # Subtract max for stability
      expscores = np.exp(scores) # (N x C)

      # nazivnik sofmaksa
      sumexp = np.sum(expscores, axis = 1, keepdims=True) # (N x 1)

      # logaritmirane vjerojatnosti razreda 
      probs = expscores / sumexp # (N x C)
      logprobs = np.log(probs) # (N x C)

      # gubitak
      loss = -np.sum(I_Y * logprobs) / N # scalar
      
      # dijagnostički ispis
      if i % 10 == 0:
        print("iteration {}: loss {}".format(i, loss))

      # derivacije komponenata gubitka po mjerama 
      dL_ds = probs - I_Y # (N x C)

      # gradijenti parametara
      grad_W = np.dot(dL_ds.T, X) / N # (C x D)
      grad_b = (np.sum(dL_ds, axis = 0) / N).reshape(-1 ,1) # C x 1 (ili 1 x C)

      # poboljšani parametri
      W += -param_delta * grad_W
      b += -param_delta * grad_b

      W_history.append(W.copy())
      b_history.append(b.copy())
    return W, b, W_history, b_history

def logreg_classify(X, W, b):
  scores = (np.dot(X, W.T) + b.T) # N x C
  scores -= np.max(scores, axis = 1, keepdims=True)  # Subtract max for stability
  expscores = np.exp(scores) # N x C
  sumexp = np.sum(expscores, axis = 1, keepdims=True) # N x 1
  probs = expscores / sumexp # N x C
  return probs

def logreg_decfun(W, b, C):
   def classify(X):
      probs = logreg_classify(X, W, b)
      if(C == 2):
         return probs[:, 1]
      return np.argmax(probs, axis = 1)
   return classify

def eval_perf_multi(Y, Y_):
    # Determine number of classes
    C = int(max(np.max(Y), np.max(Y_)) + 1)
    
    # Initialize confusion matrix
    confusion_matrix = np.zeros((C, C), dtype=int)
    
    # Populate confusion matrix (rows: true, columns: predicted)
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

def animate_graph_surface(X, Y_, W_history, b_history, bbox, interval=50, useProbs=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot()
    

    xx, yy = np.meshgrid(
        np.linspace(bbox[0][0], bbox[1][0], 100),
        np.linspace(bbox[0][1], bbox[1][1], 100)
    )
    X_ = (np.stack((xx.flatten(), yy.flatten()))).T


    # Plot initial data points
    data.graph_data(X, Y_, np.argmax(logreg_classify(X, W_history[0], b_history[0]), axis=1))
    
    offset = 0.5
    range_half = max(abs(np.min(Y_) - offset), abs(np.max(Y_) - offset))
    vmin = offset - range_half
    vmax = offset + range_half
    
    # Animation update function
    def update(frame):
        ax.clear()

        Y_pred = np.argmax(logreg_classify(X, W_history[frame], b_history[frame]), axis=1)
        probs_mesh = logreg_classify(X_, W_history[frame], b_history[frame])
        Y_mesh = np.argmax(probs_mesh, axis=1).reshape(yy.shape)

        if(not useProbs):
            mesh = ax.pcolormesh(xx, yy, Y_mesh, shading = 'auto', cmap='rainbow', alpha=0.6)
            line = ax.contour(xx, yy, Y_mesh, levels=np.arange(np.max(Y_mesh) + 1) + offset, colors='black', linewidths=0.2)
        else:
            rshpm = probs_mesh[:, 1].reshape(yy.shape)
            mesh = ax.pcolormesh(xx, yy, rshpm, vmin=vmin, vmax=vmax, shading='auto', cmap='rainbow', alpha=0.6)
            line = ax.contour(xx, yy, rshpm, levels=[offset], colors='black', linewidths=2)

        data.graph_data(X, Y_, Y_pred)
        return [mesh, line] + ax.collections
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(W_history), interval=interval, blit=False, repeat=False
    )

    # Function to handle window close event
    def on_close(event):
        plt.close('all')
    fig.canvas.mpl_connect('close_event', on_close)
    plt.show()

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset

    # instanciraj podatke X i labele Yoh_
    K = 8
    C = 2
    N = 40

    X, Y_ = data.sample_gmm_2d(K, C, N)

    Y_ = Y_.ravel()

    # train the model
    W, b, W_history, b_history = logreg_train(X, Y_, param_niter=10000, param_delta=0.01)

    # evaluate the model on the training dataset
    probs = logreg_classify(X, W, b)
    Y = np.argmax(probs, axis = 1)

    # report performance
    accuracy, cm, recall, precision = eval_perf_multi(Y, Y_)
    print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nConfusion matrix:\n {cm}")

    # graph the decision surface
    decfun = logreg_decfun(W, b, C)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    useProbs = False
    if(C == 2):
       useProbs = True
    animate_graph_surface(X, Y_, W_history, b_history, bbox, interval=50, useProbs=useProbs)
    data.graph_surface(decfun, bbox, offset=0.5, useProbs=useProbs)
    data.graph_data(X, Y_, Y)
    plt.show()
   