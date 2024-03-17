import numpy as np
import GMM
import GMMDiag
import GMMTiedCov
import metrics
import PCA
import MLlibrary
import scipy
import matplotlib.pyplot as plt

def vcol(v):
    return v.reshape((v.size, 1))
def vrow(v):
    return v.reshape((1, v.size))

priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)

numberOfComponent = 2


plt.figure()
MLlibrary.labels=[]

#Full Cov
gmm = GMM.GMM()
print("Start GMM with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(ZD, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM (numberOfComponent = 2), on application with prior", priors[i], ":", temp)
print("")
print("END")

#Naive Bayes
gmm = GMMDiag.GMMDiag()
print("Start GMM on z normalized features")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(ZD, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM on application with prior", priors[i], ":", temp)
print("")
print("END")

#Tied Cov
gmm = GMMTiedCov.GMMTiedCov()
print("Start GMM on z normalized features")
for i in range(len(priors)):
    print("")
    temp = MLlibrary.KfoldGMM(ZD, L, gmm, numberOfComponent, prior=priors[i])
    print("minDCF for GMM on application with prior", priors[i], ":", temp)
print("")
print("END")