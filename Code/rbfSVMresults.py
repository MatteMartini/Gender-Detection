# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import metrics
import PCA
import SVM
import SVMclass
import numpy as np

priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')
ZD, mean, stdv = MLlibrary.ZNormalization(D)
C = 10**(-4)
gamma = 10**(-5)
print ("Executing RBF SVM")


k = 5
named_preprocess = "RAW"


for pi in priors:
    print("--------------")
    print("Training " + str(SVM.RadialKernelBasedSvm([], None, None, None)).replace("_", "") +
          "(πT=" + str(pi) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
    rbf_svm = SVMclass.SVMclass(D, L, SVM.RadialKernelBasedSvm(priors, k, C, gamma))
    rbf_svm.make_train_with_K_fold()

named_preprocess = "Z-SCORE"


for pi in priors:
    print("--------------")
    print("Training " + str(SVM.RadialKernelBasedSvm([], None, None, None)).replace("_", "") +
          "(πT=" + str(pi) + "; C=" + str(C) + ") with " + named_preprocess + " features...")
    rbf_svm = SVMclass.SVMclass(ZD, L, SVM.RadialKernelBasedSvm(priors, k, C, gamma))
    rbf_svm.make_train_with_K_fold()