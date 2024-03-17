# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn
import metrics
import LogisticRegression

# -------------------- CONSTANTS ---------------------

classesNames = ["male", "female"]
featuresNames = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

labels=[]
SEED_VALUE = 27

# ----------------- UTILITY FUNCTIONS -----------------

def mcol(v):
    # Auxiliary function to transform 1-dim vectors to column vectors.
    return v.reshape((v.size, 1))


def mrow(v):
    # Auxiliary function to transform 1-dim vecotrs to row vectors.
    return (v.reshape(1, v.size))


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
    return U[:, ::-1][:, 0:m] #la prima parte serve per farli in ordine decrescente, perche l'operazione precedente ti da U ordinata dagli autovettori piu piccoli a quelli piu grandi!



def load(filename):
    list_of_samples = []
    list_of_labels = []
    with open(filename, 'r') as f:
        for line in f:
            data = line.split(',')
            if data[0] != '\n':
                for i in range(len(data)-1):
                    data[i] = float(data[i])
                data[-1] = int(data[-1].rstrip('\n'))
                # Now create a 1-dim array and reshape it as a column vector,
                # then append it to the appropriate list
                list_of_samples.append(mcol(np.array(data[0:-1])))
                # Append the value of the class to the appropriate list
                list_of_labels.append(data[-1])
    # We have column vectors, we need to create a matrix, so we have to
    # stack horizontally all the column vectors
    dataset_matrix = np.hstack(list_of_samples[:])
    # Create a 1-dim array with class labels
    class_label_array = np.array(list_of_labels)
    return dataset_matrix, class_label_array

def constrainSigma(sigma, psi = 0.01):

    U, s, Vh = np.linalg.svd(sigma)
    s[s < psi] = psi
    sigma = np.dot(U, mcol(s)*U.T)
    return sigma

def centerData(D):
    means = D.mean(axis=1)
    means = mcol(means)
    centeredData = D - means
    return centeredData


def custom_hist(attr_index, xlabel, D, L, classesNames):
    # Function used to plot histograms. It receives the index of the attribute to plot,
    # the label for the x axis, the dataset matrix D, the array L with the values
    # for the classes and the list of classes names (used for the legend)
    plt.hist(D[attr_index, L == 0], color="#1e90ff",
             ec="#0000ff", density=True, alpha=0.6)
    plt.hist(D[attr_index, L == 1], color="#ff8c00",
             ec="#d2691e", density=True, alpha=0.6)
    plt.legend(classesNames)
    plt.xlabel(xlabel)
    plt.show()
    return


def custom_scatter(i, j, xlabel, ylabel, D, L, classesNames):
    # Function used for scatter plots. It receives the indexes i, j of the attributes
    # to plot, the labels for x, y axes, the dataset matrix D, the array L with the
    # values for the classes and the list of classes names (used for the legend)
    plt.scatter(D[i, L == 0], D[j, L == 0], color="#1e90ff", s=10)
    plt.scatter(D[i, L == 1], D[j, L == 1], color="#ff8c00", s=10)
    plt.legend(classesNames)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    return

def plotFeatures(D, L, featuresNames, classesNames):
    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if (i == j):
                # Then plot histogram
                custom_hist(i, featuresNames[i], D, L, classesNames)
            else:
                # Else use scatter plot
                custom_scatter(
                    i, j, featuresNames[i], featuresNames[j], D, L, classesNames)
            
def plotDCF(x, y, xlabel, labella):
    plt.plot(x, y[0:len(x)], label=labella)
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    labels.append(labella)
    plt.legend(labels)
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return


def plotDCFpoly(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - c=0', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - c=1', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - c=10', color='g')
    plt.plot(x, y[3*len(x): 4*len(x)], label='min DCF prior=0.5 - c=30', color='m')

    
    plt.xlim([1e-5, 1e-1])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - c=0", "min DCF prior=0.5 - c=1", 
                'min DCF prior=0.5 - c=10', 'min DCF prior=0.5 - c=30'])

    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return

def plotDCFRBF(x, y, xlabel):
    plt.figure()
    plt.plot(x, y[0:len(x)], label='min DCF prior=0.5 - logγ=-5', color='b')
    plt.plot(x, y[len(x): 2*len(x)], label='min DCF prior=0.5 - logγ=-4', color='r')
    plt.plot(x, y[2*len(x): 3*len(x)], label='min DCF prior=0.5 - logγ=-3', color='g')
    
    plt.xlim([min(x), max(x)])
    plt.xscale("log")
    plt.legend(["min DCF prior=0.5 - logγ=-5", "min DCF prior=0.5 - logγ=-4", 
                'min DCF prior=0.5 - logγ=-3'])
    
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return


def plotDCFGMM(x, y, xlabel, labella):
    plt.plot(x, y[0:len(x)], label=labella)
    plt.xlim([min(x), max(x)])
    plt.xscale('log', base=2)
    labels.append(labella)
    plt.legend(labels)
    plt.xlabel(xlabel)
    plt.ylabel("min DCF")
    return
                
def computeAccuracy(predictedLabels, actualLabels):
    numberOfCorrectPredictions = np.array(predictedLabels == actualLabels).sum()
    accuracy = numberOfCorrectPredictions/actualLabels.size*100
    return accuracy

def computeErrorRate(predictedLabels, actualLabels):
    accuracy = computeAccuracy(predictedLabels, actualLabels)
    errorRate = 100-accuracy
    return errorRate

def Ksplit(D, L, seed=0, K=5):
    folds = []
    labels = []
    numberOfSamplesInFold = int(D.shape[1]/K)
    # Generate a random seed
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    for i in range(K):
        folds.append(D[:,idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
        labels.append(L[idx[(i*numberOfSamplesInFold): ((i+1)*(numberOfSamplesInFold))]])
    return folds, labels

def Kfold(D, L, model, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.minimum_detection_costs(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldActualDCF(D, L, model, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.compute_actual_DCF(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldActualDCFCalibrated(D, L, model, lambd=1e-4, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        scores=calibrateScores(scores, orderedLabels, lambd).flatten()
        labels = np.hstack(labels)
        return metrics.compute_actual_DCF(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldGMM(D, L, model, components, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, components)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.minimum_detection_costs(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldGMMActualDCF(D, L, model, components, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, components)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.compute_actual_DCF(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldGMMActualDCFCalibrated(D, L, model, components, lambd=1e-4, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, components)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        scores=calibrateScores(scores, orderedLabels, lambd).flatten()
        labels = np.hstack(labels)
        return metrics.compute_actual_DCF(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def fastKfoldGMM(D, L, model, components, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = [] # list of lists, the global list has a number of lists equal to the n° of components
        for k in range(components):
            scores.append([]) # initialize each sublist with an empty list. Each sublist will contain the scores for each of the 10 split
        for i in range(K):
            listOfGMMComponents = [] # it will be a list of lists of tuples
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            #Initialize D0 and D1, we will work on them
            D0 = trainingSet[:, labelsOfTrainingSet==0]
            D1 = trainingSet[:, labelsOfTrainingSet==1]
            tempList=[] # list of tuples that will contain the initial GMM estimates for class 0 and 1
            # Append the two initial estimates
            tempList.append((1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainSigma(np.cov(D0).reshape((D0.shape[0], D0.shape[0])))))
            listOfGMMComponents.append(tempList)
            tempList=[]
            tempList.append((1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainSigma(np.cov(D1).reshape((D1.shape[0], D1.shape[0])))))
            listOfGMMComponents.append(tempList) #Append the list of tuples to the list called listOfGMMComponents
            for k in range(components):
                listOfGMMComponents = model.fastTraining(trainingSet, labelsOfTrainingSet, listOfGMMComponents[0], listOfGMMComponents[1])
                scores[k].append(model.predictAndGetScores(evaluationSet))
        for k in range(components):
            scores[k] = np.hstack(scores[k])
        scores=np.array(scores).T
        orderedLabels=np.hstack(orderedLabels)
        minDCFs = []
        for k in range(components):
            minDCFs.append(metrics.minimum_detection_costs(scores[:, k], orderedLabels, prior, 1, 10))
        return minDCFs
    else:
        print("K cannot be <=1")
    return

def fastKfoldGMMDiag(D, L, model, components, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = [] # list of lists, the global list has a number of lists equal to the n° of components
        for k in range(components):
            scores.append([]) # initialize each sublist with an empty list. Each sublist will contain the scores for each of the 10 split
        for i in range(K):
            listOfGMMComponents = [] # it will be a list of lists of tuples
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            #Initialize D0 and D1, we will work on them
            D0 = trainingSet[:, labelsOfTrainingSet==0]
            D1 = trainingSet[:, labelsOfTrainingSet==1]
            tempList=[] # list of tuples that will contain the initial GMM estimates for class 0 and 1
            # Append the two initial estimates
            tempList.append((1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainSigma(np.cov(D0)*np.eye( D0.shape[0]).reshape((D0.shape[0]),D0.shape[0]))))
            listOfGMMComponents.append(tempList)
            tempList=[]
            tempList.append((1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainSigma(np.cov(D1)*np.eye( D1.shape[0]).reshape((D1.shape[0]),D1.shape[0]))))
            listOfGMMComponents.append(tempList) #Append the list of tuples to the list called listOfGMMComponents
            for k in range(components):
                listOfGMMComponents = model.fastTraining(trainingSet, labelsOfTrainingSet, listOfGMMComponents[0], listOfGMMComponents[1])
                scores[k].append(model.predictAndGetScores(evaluationSet))
        for k in range(components):
            scores[k] = np.hstack(scores[k])
        scores=np.array(scores).T
        orderedLabels=np.hstack(orderedLabels)
        minDCFs = []
        for k in range(components):
            minDCFs.append(metrics.minimum_detection_costs(scores[:, k], orderedLabels, prior, 1, 10))
        return minDCFs
    else:
        print("K cannot be <=1")
    return

def fastKfoldGMMTied(D, L, model, components, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = [] # list of lists, the global list has a number of lists equal to the n° of components
        for k in range(components):
            scores.append([]) # initialize each sublist with an empty list. Each sublist will contain the scores for each of the 10 split
        for i in range(K):
            listOfGMMComponents = [] # it will be a list of lists of tuples
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            #Initialize D0 and D1, we will work on them
            D0 = trainingSet[:, labelsOfTrainingSet==0]
            D1 = trainingSet[:, labelsOfTrainingSet==1]
            sigma0 =  np.cov(D0).reshape((D0.shape[0], D0.shape[0]))
            sigma1 =  np.cov(D1).reshape((D1.shape[0], D1.shape[0]))
            sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*sigma0+D[:, L == 1].shape[1]*sigma1)
            tempList=[] # list of tuples that will contain the initial GMM estimates for class 0 and 1
            # Append the two initial estimates
            tempList.append((1.0, D0.mean(axis=1).reshape((D0.shape[0], 1)), constrainSigma(sigma)))
            listOfGMMComponents.append(tempList)
            tempList=[]
            tempList.append((1.0, D1.mean(axis=1).reshape((D1.shape[0], 1)), constrainSigma(sigma)))
            listOfGMMComponents.append(tempList) #Append the list of tuples to the list called listOfGMMComponents
            for k in range(components):
                listOfGMMComponents = model.fastTraining(trainingSet, labelsOfTrainingSet, listOfGMMComponents[0], listOfGMMComponents[1])
                scores[k].append(model.predictAndGetScores(evaluationSet))
        for k in range(components):
            scores[k] = np.hstack(scores[k])
        scores=np.array(scores).T
        orderedLabels=np.hstack(orderedLabels)
        minDCFs = []
        for k in range(components):
            minDCFs.append(metrics.minimum_detection_costs(scores[:, k], orderedLabels, prior, 1, 10))
        return minDCFs
    else:
        print("K cannot be <=1")
    return

def KfoldSVM(D, L, model, option, c=0, d=2, gamma=1.0, C=1.0, K=5, prior=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, option = option, 
                        C=C, c = c, d=d, gamma = gamma)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.minimum_detection_costs(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return


def KfoldLR(D, L, model, lambd, K=5, prior=0.5, pi_T=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, lambd, pi_T)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.minimum_detection_costs(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldLRActualDCF(D, L, model, lambd, K=5, prior=0.5, pi_T=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, lambd, pi_T)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        labels = np.hstack(labels)
        return metrics.compute_actual_DCF(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def KfoldLRActualDCFCalibrated(D, L, model, lambd, lambd2=1e-4, K=5, prior=0.5, pi_T=0.5):
    if (K>1):
        folds, labels = Ksplit(D, L, seed=0, K=K)
        orderedLabels = []
        scores = []
        for i in range(K):
            trainingSet = []
            labelsOfTrainingSet = []
            for j in range(K):
                if j!=i:
                    trainingSet.append(folds[j])
                    labelsOfTrainingSet.append(labels[j])
            evaluationSet = folds[i]
            orderedLabels.append(labels[i])
            trainingSet=np.hstack(trainingSet)
            labelsOfTrainingSet=np.hstack(labelsOfTrainingSet)
            model.train(trainingSet, labelsOfTrainingSet, lambd, pi_T)
            scores.append(model.predictAndGetScores(evaluationSet))
        scores=np.hstack(scores)
        orderedLabels=np.hstack(orderedLabels)
        scores=calibrateScores(scores, orderedLabels, lambd2).flatten()
        labels = np.hstack(labels)
        return metrics.compute_actual_DCF(scores, orderedLabels, prior, 1, 10)
    else:
        print("K cannot be <=1")
    return

def confusionMatrix(predictedLabels, actualLabels, K):
    # Initialize matrix of K x K zeros
    matrix = np.zeros((K, K)).astype(int)
    # We're computing the confusion
    # matrix which "counts" how many times we get prediction i when the actual
    # label is j.
    for i in range(actualLabels.size):
        matrix[predictedLabels[i], actualLabels[i]] += 1
    return matrix

def Gaussianization(TD, D):
    if (TD.shape[0]!=D.shape[0]):
        print("Datasets not aligned in dimensions")
    ranks=[]
    for j in range(D.shape[0]):
        tempSum=0
        for i in range(TD.shape[1]):
            tempSum+=(D[j, :]<TD[j, i]).astype(int)
        tempSum+=1
        ranks.append(tempSum/(TD.shape[1]+2))
    y = norm.ppf(ranks)
    return y

def ZNormalization(D, mean=None, standardDeviation=None):
    if (mean is None and standardDeviation is None):
        mean = D.mean(axis=1)
        standardDeviation = D.std(axis=1)
    ZD = (D-mcol(mean))
    return ZD, mean, standardDeviation

def heatmap(D, L):
    plt.figure()
    seaborn.heatmap(np.corrcoef(D), linewidth=0.2, cmap="Greys", square=True, cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==0]), linewidth=0.2, cmap="Reds", square=True,cbar=False)
    plt.figure()
    seaborn.heatmap(np.corrcoef(D[:, L==1]), linewidth=0.2, cmap="Blues", square=True, cbar=False)
    return

def split_db_singleFold(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)  

def bayesErrorPlot(dcf, mindcf, effPriorLogOdds, model):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF", model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    return

def bayesErrorPlotV2(dcf0, dcf1, mindcf, effPriorLogOdds, model, lambda0, lambda1):
    plt.figure()
    plt.plot(effPriorLogOdds, dcf0, label='act DCF', color='r')
    plt.plot(effPriorLogOdds, dcf1, label='act DCF', color='g')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b', linestyle="--")
    plt.xlim([min(effPriorLogOdds), max(effPriorLogOdds)])
    plt.legend([model + " - act DCF lambda = "+lambda0, model + " - act DCF lambda = "+lambda1, model+" - min DCF"])
    plt.xlabel("prior log-odds")
    plt.ylabel("DCF")
    return

def calibrateScores(s, L, lambd, prior=0.5):
    # f(s) = as+b can be interpreted as the llr for the two class hypothesis
    # class posterior probability: as+b+log(pi/(1-pi)) = as +b'
    s=mrow(s)
    lr = LogisticRegression.LogisticRegression()
    lr.train(s, L, lambd, prior=prior)
    alpha = lr.x[0]
    betafirst = lr.x[1]
    calibScores = alpha*s+betafirst-np.log(prior/(1-prior))
    return calibScores

def computeOptimalBayesDecisionBinaryTaskTHRESHOLD(pi1, Cfn, Cfp, llrs, labels, t):
    predictedLabels = (llrs > t).astype(int)
    # Compute the confusion matrix
    m = confusionMatrix(predictedLabels, labels, 2)
    return m

def computeFPRTPR(pi1, Cfn, Cfp, confMatrix):
    # Compute FNR and FPR
    FNR = confMatrix[0][1]/(confMatrix[0][1]+confMatrix[1][1])
    TPR = 1-FNR
    FPR = confMatrix[1][0]/(confMatrix[0][0]+confMatrix[1][0])
    return (FPR, TPR)

def plotROC(FPR, TPR, FPR1, TPR1, FPR2, TPR2):
    # Function used to plot TPR(FPR)
    plt.figure()
    plt.grid(linestyle='--')
    plt.plot(FPR, TPR, linewidth=2, color='r')
    plt.plot(FPR1, TPR1, linewidth=2, color='b')
    plt.plot(FPR2, TPR2, linewidth=2, color='g')
    plt.legend(["MVG Tied-Cov", "Logistic regression", "GMM Full-Cov 4 components"])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    return

def build_conf_mat_uniform(prediction, L):
    conf_mat = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            conf_mat[i][j] = (1 * np.bitwise_and(prediction == i, L == j)).sum()

    return conf_mat


def compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn):
    FNR = conf_mat[0][1] / (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0] / (conf_mat[1][0] + conf_mat[0][0])
    return (pi * C_fn * FNR + (1-pi) * C_fp * FPR) / min([pi * C_fn, (1-pi) * C_fp])


def build_conf_mat(llr: np.ndarray,L: np.ndarray,pi:float, C_fn:float,C_fp:float):
    t = -np.log(pi*C_fn/((1-pi)*C_fp))
    predictions = 1*(llr > t)
    return build_conf_mat_uniform(predictions,L)


def compute_DCF(llr: np.ndarray, L: np.ndarray, pi: float, C_fn: float, C_fp: float):
    conf_mat = build_conf_mat(llr, L, pi, C_fn, C_fp)
    FNR = conf_mat[0][1]/ (conf_mat[0][1] + conf_mat[1][1])
    FPR = conf_mat[1][0]/ (conf_mat[1][0] + conf_mat[0][0])
    return pi * C_fn * FNR + (1-pi) * C_fp * FPR


def compute_NDCF(llr: np.ndarray, L: np.ndarray, pi: float, C_fn: float, C_fp: float):
    return compute_DCF(llr, L, pi, C_fn, C_fp) / min([pi*C_fn, (1-pi)*C_fp])


def compute_minimum_NDCF(llr, L, pi, C_fp, C_fn):
    llr = llr.ravel()
    tresholds = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    DCF = np.zeros(tresholds.shape[0])
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        conf_mat = build_conf_mat_uniform(pred, L)
        DCF[idx] = compute_NDCF_conf_mat(conf_mat, pi, C_fp, C_fn)
    argmin = DCF.argmin()
    return DCF[argmin], tresholds[argmin]


def compute_roc_points(llr, L):
    tresholds = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    N_label0 = (L == 0).sum()
    N_label1 = (L == 1).sum()
    ROC_points_TPR = np.zeros(L.shape[0] + 2)
    ROC_points_FPR = np.zeros(L.shape[0] + 2)
    for (idx, t) in enumerate(tresholds):
        pred = 1 * (llr > t)
        TPR = np.bitwise_and(pred == 1, L == 1).sum() / N_label1
        FPR = np.bitwise_and(pred == 1, L == 0).sum() / N_label0
        ROC_points_TPR[idx] = TPR
        ROC_points_FPR[idx] = FPR
    return ROC_points_TPR, ROC_points_FPR


def compute_det_points(llr, L):
    threshold = np.concatenate([np.array([-np.inf]), np.sort(llr), np.array([np.inf])])
    FNR_points = np.zeros(L.shape[0] + 2)
    FPR_points = np.zeros(L.shape[0] + 2)
    for (idx, t) in enumerate(threshold):
        pred = 1 * (llr > t)
        FNR = 1 - (np.bitwise_and(pred == 1, L == 1).sum() / (L == 0).sum())
        FPR = np.bitwise_and(pred == 1, L == 0).sum() / (L == 1).sum()
        FNR_points[idx] = FNR
        FPR_points[idx] = FPR
    return FNR_points, FPR_points
