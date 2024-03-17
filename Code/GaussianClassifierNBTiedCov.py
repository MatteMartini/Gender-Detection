# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:14:54 2023

@authors: Gabriele Lucca, Matteo Martini
"""


import MLlibrary
import numpy as np
import multivariateGaussian
import numpy.matlib 



class GaussianClassifierNBTiedCov:
    
    def train (self, D, L):
         self.mean0 = MLlibrary.mcol(D[:, L == 0].mean(axis=1))
         self.mean1 = MLlibrary.mcol(D[:, L == 1].mean(axis=1))
        
         
         
        
         I=np.matlib.identity(D.shape[0])
         self.sigma0 = np.multiply(np.cov(D[:, L == 0]),I)
         self.sigma1 = np.multiply(np.cov(D[:, L == 1]),I)
 
         self.sigma = 1/(D.shape[1])*(D[:, L == 0].shape[1]*self.sigma0+D[:, L == 1].shape[1]*self.sigma1)
       
         
         #class priors
         self.pi0 = D[:, L==0].shape[1]/D.shape[1]
         self.pi1 = D[:, L==1].shape[1]/D.shape[1]
       

       
         
     
    def predict (self, X):
        LS0 = multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma )
        LS1 = multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma )
        LS = np.vstack((LS0, LS1))
        
        #Log SJoints, that is the joint log-probabilities for a given sample
        LSJoint =  multivariateGaussian.joint_log_density(LS, MLlibrary.mcol(np.array([np.log(self.pi0), np.log(self.pi1) ])))
        
        #marginal log densities
        MLD = multivariateGaussian.marginal_log_densities(LSJoint)
        
        #Log-posteriors
        LP = multivariateGaussian.log_posteriors(LSJoint, MLD)
       
        
        return  np.argmax(LP, axis=0)
    
    def predictAndGetScores (self, X):
        LS0 = np.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean0, self.sigma0 )).flatten()
        LS1 = np.asarray(multivariateGaussian.logpdf_GAU_ND(X, self.mean1, self.sigma1 )).flatten()
        
        #log-likelihood ratios
        llr = LS1-LS0
        return llr
    
