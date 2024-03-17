# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 12:19:57 2023

@authors: Gabriele Lucca, Matteo Martini
"""

import LogisticRegression
import PCA
import GaussianClassifierTiedCov
import GaussianClassifier
import GMM
import numpy as np
import MLlibrary

prior=0.5
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)

DT, LT = MLlibrary.load('Test.txt')
ZDT, mean, standardDeviation = MLlibrary.ZNormalization(DT)


mvg = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
lr = LogisticRegression.LogisticRegression()
gmm = GMM.GMM()

lambd = 0
numberOfSplitToPerform = 2


print("Start Tied-Cov MVG")
FPR = []
TPR = []
mvg.train(ZD, L)
un_scores = mvg.predictAndGetScores(ZDT)
scores = MLlibrary.calibrateScores(un_scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR.append(FPRtemp)
    TPR.append(TPRtemp)
print("End Full-Cov")


print("Start Linear Logistic Regression")
FPR1 = []
TPR1 = []
lr.train(D, L, lambd, prior)
scores = lr.predictAndGetScores(DT)
scores = MLlibrary.calibrateScores(scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR1.append(FPRtemp)
    TPR1.append(TPRtemp)
print("End logistic regression")


print("Start Full-Cov 4 GMM components")
FPR2 = []
TPR2 = []
gmm.train(ZD, L, numberOfSplitToPerform)
scores = gmm.predictAndGetScores(ZDT)
scores = MLlibrary.calibrateScores(scores, LT, 1e-2).flatten()
sortedScores=np.sort(scores)
for t in sortedScores:
    m = MLlibrary.computeOptimalBayesDecisionBinaryTaskTHRESHOLD(
        prior, 1, 10, scores, LT, t)
    FPRtemp, TPRtemp = MLlibrary.computeFPRTPR(prior, 1, 10, m)
    FPR2.append(FPRtemp)
    TPR2.append(TPRtemp)
print("End GMM")


MLlibrary.plotROC(FPR, TPR, FPR1, TPR1, FPR2, TPR2)
