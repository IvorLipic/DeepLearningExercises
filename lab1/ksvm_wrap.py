from sklearn.svm import SVC
import data_l1, pt_logreg
import numpy as np
import matplotlib.pyplot as plt

class KSVMWrap():
    def  __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        '''
        Konstruira omotač i uči RBF SVM klasifikator
        X, Y_:           podatci i točni indeksi razreda
        param_svm_c:     relativni značaj podatkovne cijene
        param_svm_gamma: širina RBF jezgre'
        '''
        self.model = SVC(C=param_svm_c, gamma=param_svm_gamma, probability=True)
        self.model.fit(X, Y_)

    def predict(self, X):
        '''
        Predviđa i vraća indekse razreda podataka X
        '''
        return self.model.predict(X)
    
    def get_scores(self, X):
        '''
        Vraća klasifikacijske mjere
        (engl. classification scores) podataka X;
        ovo će vam trebati za računanje prosječne preciznosti.'
        '''
        return self.model.predict_proba(X)
    
    @property
    def support(self):
        """
        Indeksi podataka koji su odabrani za potporne vektore.
        """
        return self.model.support_
    
if __name__ == "__main__":
  K = 6
  C = 2
  N = 10
  X, Y_ = data_l1.sample_gmm_2d(K, C, N)

  svm = KSVMWrap(X, Y_, param_svm_c=1, param_svm_gamma='auto')

  probs = svm.get_scores(X)
  Y = svm.predict(X)

  accuracy, cm, recall, precision = pt_logreg.eval_perf_multi(Y, Y_)
  print(f"Accuracy: {accuracy}\nRecall: {recall}\nPrecision: {precision}\nConfusion matrix:\n {cm}")

  bbox=(np.min(X, axis=0), np.max(X, axis=0))
  if(C == 2):
    data_l1.graph_surface(lambda X: svm.get_scores(X)[:, 1], bbox, offset=0., useProbs=True)
  else:
    data_l1.graph_surface(svm.predict, bbox, offset=0., useProbs=False)

  data_l1.graph_data(X, Y_, Y, special=svm.support)
  plt.show()