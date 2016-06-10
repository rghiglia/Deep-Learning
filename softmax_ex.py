# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 12:17:18 2016

@author: rghiglia
"""

# Deep Learning

# We start with Logistic Classifier
# Linear although there is anon-linear transformation of the output
# Y = W*X + b
# Out = softmax(Y)
# X: (nO, nX) input nX input dimensions (say # pixels)
# Y: (nO, nY) output nY output dimensions # of classes

# Softmax function

scores = [3.0, 1.0, 0.2]

import numpy as np

def softmax(X):
    X = np.array(X)
    if len(X.shape)==1:
        return np.exp(X) / np.exp(X).sum()
    else:
        return np.exp(X) / np.exp(X).sum(axis=0)
            

print softmax(scores)

import matplotlib.pyplot as plt

x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2*np.ones_like(x)])

plt.plot(x, softmax(0.1*scores).T, linewidth=2)

