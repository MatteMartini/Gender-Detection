# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import metrics
import PCA
import matplotlib.pyplot as plt
import numpy as np

import SVMclass
import SVM


priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')

ZD, mean, stdv = MLlibrary.ZNormalization(D)
c_values = np.logspace(-5, 5, num=30)
d = 2
c = 1
print ("Executing polynomial SVM")


k = 5
named_preprocess = "RAW"

for pi in priors:
    for C in c_values:
        print("--------------")
        print("Training " + str(SVM.PolynomialSvm([], None, None, None, None)).replace("_", "") +
              "(πT=" + str(pi) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
        polynomial_svm = SVMclass.SVMclass(D, L, SVM.PolynomialSvm(priors, k, C, d, c))
        polynomial_svm.make_train_with_K_fold()

named_preprocess = "Z-SCORE"


for pi in priors:
    for C in c_values:
        print("--------------")
        print("Training " + str(SVM.PolynomialSvm([], None, None, None, None)).replace("_", "") +
              "(πT=" + str(pi) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
        polynomial_svm = SVMclass.SVMclass(ZD, L, SVM.PolynomialSvm(priors, k, C, d, c))
        polynomial_svm.make_train_with_K_fold()