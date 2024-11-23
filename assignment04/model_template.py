"""
    Example structure for fitting multiple models, feel free to modify to your liking
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


df = pd.read_csv('gen_data.csv')
cue_mapping = {1: 'Go+', 2: 'Go-', 3: 'NoGo+', 4: 'NoGo-'}  # Go+ = Go to win, Go- = go to avoid losing, NoGo+ = don't go to win, NoGo- = don't go to avoid losing


# exercise 1: plot the accuracy for each cue
for cue in cue_mapping:
    ...

# define yourself a softmax function
def softmax(x):
    ...

def model_1(data, learning_rate, beta):
    # run a q-learning model on the data, return the log-likelihood
    # parameters are learning rate and beta, the feedback sensitivity
    q = np.zeros(...)
    log_likelihood = ...

    for i in range(len(data)):
        ...

    return ...



method = 'Nelder-Mead'  # this optimization should work for the given data, but feel free to try others as well, they might be faster

# define a function to compute the BIC
def BIC(...):
    return ...


for j, learner in enumerate([model_1]):

    for i, subject in enumerate(np.unique(df.ID)):
        subject_data = ... # subset data to one subject
        subject_data = subject_data.reset_index(drop=True)  # not resetting the index can lead to issues

        if j == 0:

            # define yourself a loss for the current model
            def loss(params):
                return ...
            res = minimize(loss, ...initial_params..., bounds=..., method=method)

            # save the optimized log-likelihhod

            # save the fitted parameters

    # compute BIC


# plot learning rates of the last model


# Bonus


    

