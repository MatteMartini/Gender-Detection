# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""


import scipy.optimize as sp
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import MLlibrary
from itertools import repeat



def compute_weights(C, LTR, prior):
    bounds = np.zeros((LTR.shape[0]))
    empirical_pi_t = (LTR == 1).sum() / LTR.shape[0]
    bounds[LTR == 1] = C * prior[1] / empirical_pi_t
    bounds[LTR == 0] = C * prior[0] / (1 - empirical_pi_t)
    return list(zip(np.zeros(LTR.shape[0]), bounds))


def obj_svm_wrapper(H_hat):
    def obj_svm(alpha):
        alpha = MLlibrary.vcol(alpha)
        gradient = MLlibrary.vrow(H_hat.dot(alpha) - np.ones((alpha.shape[0], 1)))
        obj_l = 0.5 * alpha.T.dot(H_hat).dot(alpha) - alpha.T @ np.ones(alpha.shape[0])
        return obj_l, gradient

    return obj_svm


class LinearSvm():

    def __init__(self, priors, K, C):
        self.priors = priors
        self.scores = None
        self.DTR = None
        self.LTR = None
        self.DTE = None
        self.LTE = None
        self.K = K
        self.C = C
        self.params = [('K', K), ('C', C)]
        self.w = None


    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        
    def train(self):
        d_hat = np.vstack([self.DTR, np.ones(self.DTR.shape[1]) * self.K])
        g_hat = np.dot(d_hat.T, d_hat)
        z = 2 * self.LTR - 1
        h_hat = MLlibrary.vcol(z) * MLlibrary.vrow(z) * g_hat
        obj = obj_svm_wrapper(h_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            obj,
            np.zeros(self.LTR.size),
            bounds=compute_weights(self.C, self.LTR, self.priors),
            factr=1.0
        )
        self.w = np.dot(d_hat, MLlibrary.vcol(alpha) * MLlibrary.vcol(z))
        self.DTE = np.vstack([self.DTE, np.ones(self.DTE.shape[1]) * self.K])

    def compute_scores(self):
        self.scores = np.dot(self.w.T, self.DTE)

    def __str__(self):
        return "SVM_"


class PolynomialSvm():

    def __init__(self, priors, K, C, d, c):
        self.priors = priors
        self.scores = None
        self.DTR = None
        self.LTR = None
        self.DTE = None
        self.LTE = None
        self.K = K
        self.C = C
        self.c = c
        self.d = d
        self.params = [('K', K), ('C', C), ('c', c), ('d', d)]
        self.alpha = None

    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE
        
    def train(self):
        z = self.LTR * 2 - 1
        k_dtr = ((np.dot(self.DTR.T, self.DTR) + self.c) ** self.d) + (self.K ** 2)
        h_hat = MLlibrary.vcol(z) * MLlibrary.vrow(z) * k_dtr
        dual_obj = obj_svm_wrapper(h_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            dual_obj,
            np.zeros(self.DTR.shape[1]),
            bounds=compute_weights(self.C, self.LTR, self.priors),
            factr=1.0)
        self.alpha = alpha

    def compute_scores(self):
        z = self.LTR * 2 - 1
        self.scores = (
                MLlibrary.vcol(self.alpha) * MLlibrary.vcol(z) * ((self.DTR.T.dot(self.DTE) + self.c) ** self.d + self.K ** 2)).sum(0)

    def __str__(self):
        return "PolSVM_"


class RadialKernelBasedSvm():

    def __init__(self, priors, K, C, gamma):
        self.priors = priors
        self.scores = None
        self.DTR = None
        self.LTR = None
        self.DTE = None
        self.LTE = None
        self.params = [('K', K), ('C', C), ('g', gamma)]
        self.K = K
        self.C = C
        self.gamma = gamma
        self.alpha = None
        
    def set_attributes(self, DTR: np.ndarray, LTR: np.ndarray, DTE: np.ndarray, LTE: np.ndarray):
        
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = DTE
        self.LTE = LTE

    def train(self):
        z = 2 * self.LTR - 1
        kernel_dtr = np.zeros((self.DTR.shape[1], self.DTR.shape[1]))
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTR.shape[1]):
                kernel_dtr[i][j] = np.exp(
                    - self.gamma * np.linalg.norm(self.DTR[:, i] - self.DTR[:, j]) ** 2) + self.K ** 2
        h_hat = MLlibrary.vcol(z) * MLlibrary.vrow(z) * kernel_dtr
        dual_obj = obj_svm_wrapper(h_hat)
        alpha, _, _ = fmin_l_bfgs_b(
            dual_obj,
            np.zeros(self.DTR.shape[1]),
            bounds=compute_weights(self.C, self.LTR, self.priors),
            factr=1.0
        )
        self.alpha = alpha

    def compute_scores(self):
        dist = np.zeros((self.DTR.shape[1], self.DTE.shape[1]))
        z = 2 * self.LTR - 1
        for i in range(self.DTR.shape[1]):
            for j in range(self.DTE.shape[1]):
                dist[i][j] += np.exp(-self.gamma * np.linalg.norm(self.DTR[:, i:i+1] - self.DTE[:, j:j+1]) ** 2) + (self.K)**2
        self.scores = (MLlibrary.vcol(self.alpha) * MLlibrary.vcol(z) * dist).sum(0)

    def __str__(self):
        return "RadialBasedSVM_"   