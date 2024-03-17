# -*- coding: utf-8 -*-
"""

@authors: Gabriele Lucca, Matteo Martini
"""

import numpy as np
import MLlibrary

def bayes_optimal_decisions(llr, pi1, cfn, cfp):
    
    threshold = -np.log(pi1*cfn/((1-pi1)*cfp))
    predictions = (llr > threshold ).astype(int)
    return predictions


def detection_cost_function (M, pi1, cfn, cfp):
    FNR = M[0][1]/(M[0][1]+M[1][1])
    FPR = M[1][0]/(M[0][0]+M[1][0])
    
    return (pi1*cfn*FNR +(1-pi1)*cfp*FPR)

def normalized_detection_cost_function (DCF, pi1, cfn, cfp):
    dummy = np.array([pi1*cfn, (1-pi1)*cfp])
    index = np.argmin (dummy) 
    return DCF/dummy[index]

def minimum_detection_costs (llr, LTE, pi1, cfn, cfp):
    
    sorted_llr = np.sort(llr)
    
    NDCF= []
    
    for t in sorted_llr:
        predictions = (llr > t).astype(int)
        
        confMatrix =  MLlibrary.confusionMatrix(predictions, LTE, LTE.max()+1)
        uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
        NDCF.append(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
        
    index = np.argmin(NDCF)
    
    return NDCF[index]

def compute_actual_DCF(llr, LTE, pi1, cfn, cfp):
    
    predictions = (llr > (-np.log(pi1/(1-pi1)))).astype(int)
    
    confMatrix =  MLlibrary.confusionMatrix(predictions, LTE, LTE.max()+1)
    uDCF = detection_cost_function(confMatrix, pi1, cfn, cfp)
        
    NDCF=(normalized_detection_cost_function(uDCF, pi1, cfn, cfp))
        
    return NDCF

def split_data(DTR, LTR, K=5, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(DTR.shape[1])
    d = np.hsplit(DTR[:, idx], K)
    l = np.hsplit(LTR[idx], K)
    return d, l, idx
