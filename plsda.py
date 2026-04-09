from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import f1_score
import numpy as np

class PLSDA(PLSRegression):
    def __init__(
        self,
        n_components = 2,
        scale = True,
        max_iter = 500,
        tol = 1e-06,
        copy = True
    ):
        super().__init__(
            n_components = n_components,
            scale = scale,
            max_iter = max_iter,
            tol = tol,
            copy = copy
        )
    
    def reg_predict(self, X):
        return super().predict(X)
        
    def predict(self, X):
        Y_prob = super().predict(X)
        return np.argmax(Y_prob, axis = 1)
    
    def score(self, X, Y):
        Y_pred = self.predict(X)
        Y_true = np.argmax(Y, axis = 1)

        return f1_score(Y_true, Y_pred, average='macro')