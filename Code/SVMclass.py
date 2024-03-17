# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:32:47 2023

@author: gasti
"""
from itertools import compress

import metrics
import MLlibrary
import numpy as np
import scipy as sp

def debug_print_information(model, labels):
    predicted_labels = np.where(model.scores > 0, 1, 0)
    err = (1 - (labels == predicted_labels).sum() / labels.size) * 100
    print("Error rate for this training is " + str(round(err, 2)) + "%")
    cost_0_5 = str(round(MLlibrary.compute_minimum_NDCF(model.scores, labels, 0.5, 1, 1)[0], 3))
    cost_0_1 = str(round(MLlibrary.compute_minimum_NDCF(model.scores, labels, 0.1, 1, 1)[0], 3))
    cost_0_9 = str(round(MLlibrary.compute_minimum_NDCF(model.scores, labels, 0.9, 1, 1)[0], 3))
    print("minDCF with π=0.5 " + cost_0_5)
    print("minDCF with π=0.1 " + cost_0_1)
    print("minDCF with π=0.9 " + cost_0_9)

class SVMclass:

    def __init__(self, DTR: np.ndarray, LTR: np.ndarray, model):
        self.DTR = DTR
        self.LTR = LTR
        self.DTE = None
        self.LTE = None
        self.model = model


    def make_train_with_K_fold(self, K=5, seed=0):

        D, L, idx = metrics.split_data(self.DTR, self.LTR, K, seed)
        mask = np.array([False for _ in range(K)])
        scores = np.zeros(self.LTR.shape[0])
        n_folds = self.LTR.shape[0] // K
        labels_training = self.LTR[idx]


        for i in range(K):

            mask[i] = True

            DTE = np.array(list(compress(D, mask))).reshape(-1, D[0].shape[1])
            DTR = np.hstack(np.array(list(compress(D, ~mask))))
            LTE = np.array(list(compress(L, mask))).ravel()
            LTR = np.hstack(np.array(list(compress(L, ~mask))))


            # Apply the model selected for the  SVMclass with training and save the score
            self.model.set_attributes(DTR, LTR, DTE, LTE)
            self.model.train()
            self.model.compute_scores()
            scores[i * n_folds: (i + 1) * n_folds] = self.model.scores
            mask[i] = False
        self.model.scores = scores

        # Print some debug information
        debug_print_information(self.model, labels_training)


