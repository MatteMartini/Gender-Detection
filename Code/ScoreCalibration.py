import MLlibrary
import PCA
import metrics
import LogisticRegression
import GMM
import GaussianClassifierTiedCov
import GMMDiag
import GMMTiedCov
import numpy as np
import matplotlib.pyplot as plt
import scipy

def vcol(v):
    return v.reshape((v.size, 1))
def vrow(v):
    return v.reshape((1, v.size))


def SbSw(D, L):
    SB = 0
    SW = 0
    mu = vcol(D.mean(1))  #media di tutti i sample di una riga per ogni attributo, è un vettore di 4 elementi in questo caso, perche ci sono 4 possibili attributi => è la media del dataset, cioe la media di tutti i sample, distinto per ogni attributo!
    for i in range(L.max() + 1): #L.max() +1 ti da il numero di classi differenti tra i dati passati
        DCls = D[:, L == i]  #ti prendi cosi tutti i sample della classe in analisi, e saranno ad classe 0, poi classe 1 ecc.
        muCls = vcol(DCls.mean(1)) #media degli elementi di una classe! Grazie alla riga prima escludi gli elementi della classe in analisi
        SW += np.dot(DCls - muCls, (DCls - muCls).T)  #ad ogni iterazione aggiungi il contributo della parte a destra
        SB += DCls.shape[1] * np.dot(muCls - mu, (muCls - mu).T)  #DCls.shape[1] corrisponde al nc della formula

    SW /= D.shape[1] # è il fratto N che sta nelle 2 formule
    SB /= D.shape[1]
    return SB, SW

def LDA(D, L, m):
    SB, SW = SbSw(D, L)
    s, U = scipy.linalg.eigh(SB, SW) 
    return U[:, ::-1][:, 0:m]


priors = [0.1, 0.5, 0.9]
D, L = MLlibrary.load('Train.txt')
ZD, mean, standardDeviation = MLlibrary.ZNormalization(D)

#Models
gc = GaussianClassifierTiedCov.GaussianClassifierTiedCov()
lr = LogisticRegression.LogisticRegression()
gmm = GMM.GMM()

#Hyperparameters
lambd = 0
C = 5
gamma = 0.1
numberOfSplitToPerform = 2

numberOfPoints=13
effPriorLogOdds = np.linspace(-6, 6, numberOfPoints)
effPriors = 1/(1+np.exp(-1*effPriorLogOdds))

print("Start Tied-Cov") 
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldActualDCF(ZD, L, gc, prior=priors[i])
    print("Actual DCF MVG Tied-cov with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")


print("Start Linear Logistic Regression (lambda=0)")  
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldLRActualDCF(ZD, L, lr, lambd, prior=priors[i])
    print("Actual DCF Linear Logistic Regression with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")


print("Start GMM 4 components") 
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldGMMActualDCF(ZD, L, gmm, numberOfSplitToPerform, prior=priors[i])
    print("Actual DCF Full-Cov 8 GMM components with prior=%.1f:  %.3f" %(priors[i], actualDCF))
print("")
print("END")

#FULL-COV RAW
actualDCFs = []
minDCFs = []
for i in range(numberOfPoints):
    actualDCFs.append(MLlibrary.KfoldActualDCF(ZD, L, gc, prior=effPriors[i]))
    minDCFs.append(MLlibrary.Kfold(ZD, L, gc, prior=effPriors[i]))
    print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
MLlibrary.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Full-Cov")

# Linear Logistic Regression
actualDCFs = []
minDCFs = []
for i in range(numberOfPoints):
    actualDCFs.append(MLlibrary.KfoldLRActualDCF(ZD, L, lr, lambd, prior=effPriors[i]))
    minDCFs.append(MLlibrary.KfoldLR(ZD, L, lr, lambd, prior=effPriors[i]))
    print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
MLlibrary.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "Logistic Regression")


#GMM Z-Norm 4 components 
actualDCFs = []
minDCFs = []
for i in range(numberOfPoints):
    actualDCFs.append(MLlibrary.KfoldGMMActualDCF(ZD, L, gmm, numberOfSplitToPerform, prior=effPriors[i]))
    minDCFs.append(MLlibrary.KfoldGMM(ZD, L, gmm, numberOfSplitToPerform, prior=effPriors[i]))
    print("At iteration", i, "the min DCF is", minDCFs[i], "and the actual DCF is", actualDCFs[i])
MLlibrary.bayesErrorPlot(actualDCFs, minDCFs, effPriorLogOdds, "GMM")


#Score calibration on Tied Cov
actualDCFs0 = []
actualDCFs1 = []
minDCFs = []
for i in range(numberOfPoints):
    print("Working on point:", i)
    minDCFs.append(MLlibrary.Kfold(ZD, L, gc, prior=effPriors[i]))
    actualDCFs0.append(MLlibrary.KfoldActualDCFCalibrated(ZD, L, gc, lambd=1e-3, prior=effPriors[i]))
    actualDCFs1.append(MLlibrary.KfoldActualDCFCalibrated(ZD, L, gc, lambd=1e-2, prior=effPriors[i]))
MLlibrary.bayesErrorPlotV2(actualDCFs0, actualDCFs1, minDCFs, effPriorLogOdds, "Full Cov", "10^(-3)", "10^(-2)")



#Score calibration on Logistic Regression
actualDCFs0 = []
actualDCFs1 = []
minDCFs = []
for i in range(numberOfPoints):
    print("Working on point:", i)
    minDCFs.append(MLlibrary.KfoldLR(ZD, L, lr, lambd, prior=effPriors[i]))
    actualDCFs0.append(MLlibrary.KfoldLRActualDCFCalibrated(ZD, L, lr, lambd, lambd2=1e-3, prior=effPriors[i]))
    actualDCFs1.append(MLlibrary.KfoldLRActualDCFCalibrated(ZD, L, lr, lambd, lambd2=1e-2, prior=effPriors[i]))
MLlibrary.bayesErrorPlotV2(actualDCFs0, actualDCFs1, minDCFs, effPriorLogOdds, "Logistic Regression", "10^(-3)", "10^(-2)")



#Score calibration on GMM Z-Norm 4 components
actualDCFs0 = []
actualDCFs1 = []
minDCFs = []
for i in range(numberOfPoints):
    print("Working on point:", i)
    minDCFs.append(MLlibrary.KfoldGMM(ZD, L, gmm, numberOfSplitToPerform, prior=effPriors[i]))
    actualDCFs0.append(MLlibrary.KfoldGMMActualDCFCalibrated(ZD, L, gmm, numberOfSplitToPerform, lambd=1e-3, prior=effPriors[i]))
    actualDCFs1.append(MLlibrary.KfoldGMMActualDCFCalibrated(ZD, L, gmm, numberOfSplitToPerform, lambd=1e-2, prior=effPriors[i]))
MLlibrary.bayesErrorPlotV2(actualDCFs0, actualDCFs1, minDCFs, effPriorLogOdds, "GMM", "10^(-3)", "10^(-2)")



print("Start Tied-Cov SCORES CALIBRATED")
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldActualDCFCalibrated(ZD, L, gc, lambd, prior=priors[i])
    print("Actual DCF Full-Cov with prior=%.1f:  %.3f after score calibration" %(priors[i], actualDCF))
print("")
print("END")


print("Start Linear Logistic Regression SCORES CALIBRATED")
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldLRActualDCFCalibrated(ZD, L, lr, lambd, lambd2=1e-2, prior=priors[i])
    print("Actual DCF Linear Logistic Regression with prior=%.1f:  %.3f after score calibration" %(priors[i], actualDCF))
print("")
print("END")


print("Start  Naive Bayes 4 GMMDiag components SCORES CALIBRATED")
print("")
for i in range(len(priors)):
    actualDCF = MLlibrary.KfoldGMMActualDCFCalibrated(ZD, L, gmm, numberOfSplitToPerform, lambd, prior=priors[i])
    print("Actual Naive Bayes 4 GMMDiag components with prior=%.1f:  %.3f after score calibration" %(priors[i], actualDCF))
print("")
print("END")