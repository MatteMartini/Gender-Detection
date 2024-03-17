import numpy as np
import scipy.optimize
import LogRegFunctions
import LogisticRegression
import MLlibrary
import metrics
import matplotlib.pyplot as plt
import PCA
  
priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
lambd = 0
lr = LogisticRegression.LogisticRegression()

print("Start logistic regression with 5-fold on RAW features")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(D, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=0, pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")


print("Start logistic regression with 5-fold on z normalized features")
for j in range(len(priors)):
    print("")
    for i in range(len(priors)):
        temp = MLlibrary.KfoldLR(ZD, L, lr, lambd, prior=priors[i], pi_T=priors[j])
        print("minDCF for Log-Reg (lambda=0, pi_T =", priors[j], ") on application with prior", priors[i], ":", temp)
print("")
print("END")
