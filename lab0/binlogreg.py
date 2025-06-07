import numpy as np
import data
import matplotlib.pyplot as plt
from matplotlib import animation

def binlogreg_train(X,Y_, param_niter, param_delta):
  '''
    Argumenti
      X:  podatci, np.array NxD
      Y_: indeksi razreda, np.array Nx1

    Povratne vrijednosti
      w, b: parametri logističke regresije
  ''' 
  Y_ = Y_.reshape(-1 ,1)
  N = X.shape[0]
  D = X.shape[1]

  w = np.random.randn(D, 1) * 3
  b = np.zeros((1, 1))

  w_history = []

  # gradijentni spust (param_niter iteracija)
  for i in range(param_niter):
    # klasifikacijske mjere
    scores = np.dot(X, w) + b # (N x D) x (D x 1) --> (N x 1)

    # vjerojatnosti razreda c_1
    probs = 1 / (1 + np.e**(-scores)) # (N x 1)

    # gubitak
    loss  = -np.sum(Y_ * np.log(probs) + (np.ones(N) - Y_) * np.log(1 - probs)) / N # scalar

    w_history.append(np.concatenate([w.flatten(), b.flatten()]))

    # dijagnostički ispis
    if i % 10 == 0:
      print("iteration {}: loss {}".format(i, loss))

    # derivacije gubitka po klasifikacijskim mjerama
    dL_dscores = probs - Y_ # (N x 1)
    
    # gradijenti parametara
    grad_w = np.dot(X.T, dL_dscores) / N # (D x N) x (N x 1) --> (D x 1)
    grad_b = np.sum(dL_dscores) / N # scalar

    # poboljšani parametri
    w += -param_delta * grad_w # (D x 1) + (D x 1)
    b += -param_delta * grad_b # scalar + scalar
    
  return np.array(w_history), w, b

def binlogreg_classify(X, w, b):
    '''
    Argumenti
        X:    podatci, np.array NxD
        w, b: parametri logističke regresije 

    Povratne vrijednosti
        probs: vjerojatnosti razreda c1
    '''
    scores = np.dot(X, w) + b # (N x 1)
    probs = 1 / (1 + np.e**(-scores)) # (N x 1)
    return probs

def binlogreg_decfun(w,b):
    def classify(X):
        return binlogreg_classify(X, w, b)
    return classify

#################################
def loss_function(w1, w2, b, X, Y_):
    w = np.array([w1, w2])
    scores = np.dot(X, w) + b
    probs = np.clip(1 / (1 + np.e**(-scores)), 1e-12, 1 - 1e-12)
    loss = -np.sum(Y_ * np.log(probs) + (1 - Y_) * np.log(1 - probs))
    return loss


def visualize_loss_func(X, Y_, w_history):
  w1_values = np.linspace(-5, 5, 100)
  w2_values = np.linspace(-5, 5, 100)
  W1, W2 = np.meshgrid(w1_values, w2_values)
  loss_values = np.zeros_like(W1)

  for i in range(W1.shape[0]): ### Using last recorded w0!
      for j in range(W1.shape[1]):
          loss_values[i, j] = loss_function(W1[i, j], W2[i, j], w_history[-1, 2], X, Y_)

  fig = plt.figure(figsize=(12, 6))

  # 3D plot
  ax1 = fig.add_subplot(121, projection='3d')
  ax1.plot_surface(W1, W2, loss_values, cmap='inferno', alpha = 0.8)
  ax1.set_xlabel('w1')
  ax1.set_ylabel('w2')
  ax1.set_zlabel('Loss')
  ax1.set_title('3D ploha funkcije gubitka')

  # Konturni dijagram
  ax2 = fig.add_subplot(122)
  contour = ax2.contourf(W1, W2, loss_values, levels=50, cmap='inferno')
  plt.colorbar(contour)
  ax2.set_xlabel('w1')
  ax2.set_ylabel('w2')
  ax2.set_title('Konturni dijagram funkcije gubitka')

  ## Putanje
  # 2d
  line2d, = ax2.plot([], [], 'r.-', label = 'Putanja optimizacije')
  point2d, = ax2.plot([], [], 'bo', label = 'Trenutna točka')
  ax2.legend()
  # 3d
  line3d, = ax1.plot([], [], [], 'r.-', label='Putanja optimizacije')
  point3d, = ax1.plot([], [], [], 'bo', label='Trenutna točka')
  ax1.legend()

  # Funkcija za inicijalizaciju animacije
  def init():
    line2d.set_data([], [])
    point2d.set_data([], [])

    line3d.set_data([], [])
    line3d.set_3d_properties([])
    point3d.set_data([], [])
    point3d.set_3d_properties([])

    return line2d, point2d, line3d, point3d

  # Ažuriranje animacije
  def update(frame):
    line2d.set_data(w_history[:frame, 0], w_history[:frame, 1])
    point2d.set_data([w_history[frame, 0]], [w_history[frame, 1]])

    w_current = w_history[frame, :2].reshape(-1, 1)
    b_current = w_history[frame, 2].reshape(-1, 1)
    loss_current = loss_function(w_current[0], w_current[1], b_current, X, Y_)
    line3d.set_data(w_history[:frame, 0], w_history[:frame, 1])
    line3d.set_3d_properties([loss_function(w_history[i, 0], w_history[i, 1], w_history[-1, 2], X, Y_) for i in range(frame)])
    point3d.set_data([w_history[frame, 0]], [w_history[frame, 1]])
    point3d.set_3d_properties([loss_current])

    return line2d, point2d, line3d, point3d

  # Stvaranje animacije
  anim = animation.FuncAnimation(
    fig, update, frames=len(w_history), init_func=init, blit=True, interval=50, repeat=False
  )

  plt.show()

  #################################

if __name__=="__main__":
    #np.random.seed(100)

    # get the training dataset
    X,Y_ = data.sample_gmm_2d(K = 6, C = 2, N = 20)

    # train the model
    w_history, w, b = binlogreg_train(X, Y_, 1000, 0.01)

    # evaluate the model on the training dataset
    probs = binlogreg_classify(X, w, b)
    Y = (probs >= 0.5).astype(int)

    # report performance
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probs.flatten().argsort()])
    print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nAP: {AP}")

    visualize_loss_func(X, Y_, w_history)

    # graph the decision surface
    decfun = binlogreg_decfun(w,b)
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(decfun, bbox, offset=0.5, useProbs=True)
    
    # graph the data points
    data.graph_data(X, Y_.ravel(), Y.ravel())

    # show the plot
    plt.show()






