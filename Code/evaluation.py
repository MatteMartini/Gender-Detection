# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 01:29:05 2023

@authors: Gabriele Lucca, Matteo Martini
"""

import MLlibrary
import GaussianClassifier
import GaussianClassifierNB
import GaussianClassifierTiedCov
import LogisticRegression
import SVMclass
import GMM
import GMMTiedCov
import GMMDiag

import metrics
import matplotlib.pyplot as plt
import numpy as np

priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')
DT, LT = MLlibrary.load('Test.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
ZDT, mean, standardDeviation = MLlibrary.ZNormalization(DT)

print ("----MVG Full Cov----")
model = GaussianClassifier.GaussianClassifier()
model.train(ZD, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF MVG Full-Cov with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
   
print ("----MVG Diag Cov----")
model = GaussianClassifierNB.GaussianClassifierNB()
model.train(ZD, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF MVG Diag Cov with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
  
print ("----MVG Tied Cov----")
model = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
model.train(ZD, L)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF MVG Tied-Cov with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
   
lambd=0
print ("----Logistic Regression----")
model = LogisticRegression.LogisticRegression()
model.train(ZD, L, lambd, 0.5)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF Logistic Regression with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=10
print ("----Linear SVM----")
model = SVM.SVM()
model.train (ZD, L, option='linear', C=C)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF Linear SVM with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

C=10
print ("----Quadratic SVM----")
model = SVM.SVM()
model.train (ZD, L, option='polynomial', d=2, c=1, C=C)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF Quadratic SVM with prior=%.1f:  %.3f" %(priors[i], minDCFSF))


C=10
print ("----RBF SVM----")
model = SVMclass.SVMclass()
model.train (ZD, L, option='RBF', C=C, gamma=0.001)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1)  
    print("min DCF RBF SVM with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 2 components")
model = GMM.GMM()
model.train(ZD, L, 1)
for i in range(len(priors)):
    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    print("min DCF of GMM Full-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 4 components")
model = GMM.GMM()
model.train(ZD, L, 2)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Full-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Full Cov 8 components")
model = GMM.GMM()
model.train(ZD, L, 3)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Full-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 2 components")
model = GMMTiedCov.GMMTiedCov()
model.train(ZD, L, 1)
# GMM with Tied Cov 2 components
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Tied-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 4 components")
model = GMMTiedCov.GMMTiedCov()
model.train(ZD, L, 2)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Tied-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Tied Cov 8 components")
model = GMMTiedCov.GMMTiedCov()
model.train(ZD, L, 3)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Tied-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 2 components")
model = GMMDiag.GMMDiag()
model.train(ZD, L, 1)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Diag-Cov model with 2 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 4 components")
model = GMMDiag.GMMDiag()
model.train(ZD, L, 2)
for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Diag-Cov model with 4 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))

print ("GMM with Diag Cov 8 components")
model = GMMDiag.GMMDiag()
model.train(ZD, L, 3)

for i in range(len(priors)):

    minDCFSF=metrics.minimum_detection_costs(model.predictAndGetScores(ZDT), LT, priors[i], 1,1) 
    
    print("min DCF of GMM Diag-Cov model with 8 components with prior=%.1f:  %.3f" %(priors[i], minDCFSF))
