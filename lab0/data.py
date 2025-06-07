import matplotlib.pyplot as plt
import numpy as np

class Random2DGaussian:
    def __init__(self):
        minx = 0
        maxx = 10
        miny = 0
        maxy = 10
        dx = maxx - minx
        dy = maxy - miny
        self.mu = np.random.random_sample(2)*(dx, dy) # Mu
        eigvals = (np.random.random_sample(2)*(dx/5, dy/5))**2
        D = np.array([[eigvals[0], 0],
                      [0, eigvals[1]]])
        angle = np.random.random_sample()*np.pi*2
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        self.cov_mat = np.dot(np.dot(R, D), R.T) # Cov matrix (D x D) -> D = 2

    def get_sample(self, n):
        return np.random.multivariate_normal(self.mu, self.cov_mat, n)

def sample_gauss_2d(C, N):
    X = np.zeros((C * N, 2))
    Y = np.zeros(C * N, dtype=int)

    for idx in range(C):
        G = Random2DGaussian()
        X[idx * N:(idx + 1) * N] = G.get_sample(N)
        Y[idx * N:(idx + 1) * N] = idx

    return X, Y

def sample_gmm_2d(K, C, N):
    classes = list(range(C))
    np.random.shuffle(classes)

    X = np.zeros((K * N, 2))
    Y = np.zeros(K * N, dtype=int)

    for k in range(K):
        G = Random2DGaussian()
        X[k * N:(k + 1) * N] = G.get_sample(N)
        Y[k * N:(k + 1) * N] = classes[k % C]

    return X, Y

def eval_perf_binary(Y, Y_):
    TP = TN = FP = FN = acc = prec = rec = 0
    for (a, b) in zip(Y, Y_):
      if(a == 1 and b == 1):
         TP += 1
      elif(a == 1 and b == 0):
         FP += 1
      elif(a == 0 and b == 1):
         FN += 1
      else:
         TN += 1
    if((TP + TN + FP + FN) != 0):
        acc = (TP + TN) / (TP + TN + FP + FN)
    if((TP + FN) != 0):
        rec = TP / (TP + FN)
    if((TP + FP) != 0):
        prec = TP / (TP + FP)
    return acc, rec, prec

def eval_AP(Y_r):
    '''
    Argumenti:
        Y_r: rangirana lista točnih razreda (0 ili 1), np.array ili lista

    Povratna vrijednost:
        AP: prosječna preciznost
    '''
    Y_r = np.array(Y_r).ravel()
    N = len(Y_r)

    cumulative_precision = 0
    num_positives = np.sum(Y_r)
    
    if num_positives == 0:
        return 0
    
    for i in range(N):
        if(Y_r[i] == 1 or Y_r[i] == True):
            y_1 = np.zeros(i)
            y_2 = np.ones(N - i)
            y = np.hstack((y_1, y_2))
            TP = FP = 0
            for (a, b) in zip(y, Y_r):
                if(a == 1 and b == 1):
                    TP += 1
                elif(a == 1 and b == 0):
                    FP += 1
            precision_at_i = TP / (TP + FP)
            cumulative_precision += precision_at_i

    AP = cumulative_precision / num_positives
    return AP

def graph_data(X, Y_, Y):    
    '''
    X  ... podatci (np.array dimenzija Nx2)
    Y_ ... točni indeksi razreda podataka (Nx1)
    Y  ... predviđeni indeksi razreda podataka (Nx1)
    ''' 
    Y = Y.ravel()
    Y_ = Y_.ravel()
    num_classes = len(np.unique(Y_))
    colors = plt.cm.rainbow(np.linspace(0, 1, num_classes))

    for i in range(num_classes):
        cls = (Y_ == i).ravel()

        base_color = colors[i]
        edge_color = np.clip(base_color - [0.01,0.01,0.01,0], 0, 1)

        plt.scatter(X[cls & (Y == Y_), 0], X[cls & (Y == Y_), 1], c=[base_color], marker='o', edgecolor='black')
        plt.scatter(X[cls & (Y != Y_), 0], X[cls & (Y != Y_), 1], facecolor='none', marker='s', edgecolor=edge_color)

def graph_surface(fun, rect, offset, width = 256, height = 256, useProbs=False):
    '''
    fun    ... decizijska funkcija (Nx2)->(Nx1)
    rect   ... željena domena prikaza zadana kao:
                ([x_min,y_min], [x_max,y_max])
    offset ... "nulta" vrijednost decizijske funkcije na koju 
                je potrebno poravnati središte palete boja;
                tipično imamo:
                offset = 0.5 za probabilističke modele 
                    (npr. logistička regresija)
                offset = 0 za modele koji ne spljošćuju
                    klasifikacijske mjere (npr. SVM)
    width,height ... rezolucija koordinatne mreže
    '''
    xx, yy = np.meshgrid(np.linspace(rect[0][0], rect[1][0], width),
                         np.linspace(rect[0][1], rect[1][1], height))
    X = (np.stack((xx.flatten(), yy.flatten()))).T
    Y = fun(X).reshape(yy.shape)
    if(useProbs):
        range_half = max(abs(np.min(Y) - offset), abs(np.max(Y) - offset))
        vmin = offset - range_half
        vmax = offset + range_half
        plt.pcolormesh(xx, yy, Y, vmin=vmin, vmax=vmax, shading='auto', cmap='rainbow', alpha=0.6)
        plt.contour(xx, yy, Y, levels=[offset], colors='black', linewidths=2)
    else:
        plt.pcolormesh(xx, yy, Y, shading='auto', cmap='rainbow', alpha=0.6)
        plt.contour(xx, yy, Y, levels=np.arange(np.max(Y) + 1) + offset, colors='black', linewidths=0.5)
        

def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores

if __name__=="__main__":
    np.random.seed(100)

    # get the training dataset
    X,Y_ = sample_gmm_2d(K = 4, C = 2, N = 30)
    #X,Y_ = sample_gauss_2d(2, 100)

    print(Y_)
    
    # get the class predictions
    Y = myDummyDecision(X) > 0.5
    
    # graph the decision surface
    graph_surface(myDummyDecision, (np.min(X, axis=0), np.max(X, axis=0)), offset = 0.5, useProbs=True) 
    
    # graph the data points
    graph_data(X, Y_, Y)

    plt.show()