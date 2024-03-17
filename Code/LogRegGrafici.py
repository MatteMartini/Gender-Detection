import MLlibrary
import metrics
import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import PCA
import scipy


if __name__ == '__main__':
 priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)
lambd = 0
lr = LogisticRegression.LogisticRegression()
lambdas=np.logspace(-4, 2.5, num=49)
plt.figure()


# #Inizio nuovo grafico con K=5 folds
plt.figure()
MLlibrary.labels=[]

minDCF5foldRaw = []
print("Start logistic regression with 5-fold on RAW features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(D, L, lr, l, prior=priors[i])
        minDCF5foldRaw.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5foldRaw, "λ", "RAW")


minDCF5foldZ = []
print("Start logistic regression with 5-fold on z normalized features")
for i in range(len(priors)):
    print("")
    print("Working on application with prior:", priors[i])
    for l in lambdas:
        temp = MLlibrary.KfoldLR(ZD, L, lr, l, prior=priors[i])
        minDCF5foldRaw.append(temp)
        print("For lambda", l, "the minDCF is", temp)
print("")
print("END")
MLlibrary.plotDCF(lambdas, minDCF5foldZ, "λ", "Z-score")
