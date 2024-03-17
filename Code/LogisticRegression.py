# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import numpy as np
import scipy.optimize
import LogRegFunctions

class LogisticRegression:
    
    def __init__(self):
        self.l = 0
        self.priors = [0.5, 0.5]

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE


    def train(self, D, L, lambd, prior=0.5):
        self.x, self.f, self.d = scipy.optimize.fmin_l_bfgs_b(LogRegFunctions.logreg_obj, np.zeros(
            D.shape[0] + 1), args=(D, L, lambd, prior), approx_grad=True)
        
    def predict(self, X):
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        predictedLabels = (scores>0).astype(int)
        return predictedLabels
    
    def predictAndGetScores(self, X):
        scores = np.dot(self.x[0:-1], X) + self.x[-1]
        return scores
    
    def train_fusion(self):
        fun_to_minimize = logreg_obj_wrap(self)
        minim, j, _ = scipy.optimize.fmin_l_bfgs_b(fun_to_minimize, np.zeros((self.DTR.shape[0] + 1)), approx_grad=True,
                                                factr=1.0)
        self.w = minim[0:-1]
        self.b = minim[-1]

    def compute_scores(self):
        self.scores = np.dot(self.w.T, self.DTE) + self.b
    
def logreg_obj_wrap(model: LogisticRegression):
    z = 2 * model.LTR - 1

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        s = 0
        const = (model.l / 2) * (np.dot(w, w.T))
        for i in range(np.unique(model.LTR).size):
            const_class = (model.priors[i] / model.LTR[model.LTR == i].size)
            s += const_class * np.logaddexp(0, -z[model.LTR == i] * (np.dot(w.T, model.DTR[:, model.LTR == i]) + b)).sum()

        return const + s

    return logreg_obj
