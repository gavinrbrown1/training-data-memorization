# Code for experiments in
# "When Is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?"
# Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar

# This file contains the code to generate data, evaluate accuracy

import numpy as np
import torch

def create_problem_instance(params):
    """Return instance, dictionary"""
    fixed_indices = []  # will contain numpy arrays
    fixed_values = []
    for i in range(params['N']):
        num_fixed = np.random.binomial(params['d'], params['rho'])  # number of fixed indices
        if num_fixed == 0:
            print('Error! Worry about this!')
        if num_fixed > params['d']:
            num_fixed = params['d']
        fixed_indices.append(np.random.choice(params['d'], size=num_fixed, replace=False))
        fixed_values.append(np.random.choice([-1,+1], size=num_fixed, replace=True))
    return {'indices': fixed_indices, 'values': fixed_values}

def sample_from_subpop(instance, params, subpop):
    """sample one point from a given subpop"""
    y = subpop
    x = np.random.choice([-1,+1], size=params['d'])
    x[instance['indices'][subpop]] = instance['values'][subpop]
    return x, y, subpop

def sample_uniform(instance, params):
    """wrapper for sample_from_subpop() with uniform subpopulation choice"""
    subpop = np.random.randint(params['N'])
    return sample_from_subpop(instance, params, subpop)

# generate test data set, train and test baseline classifier

def make_data(params, instance):

    # generate training data
    X = np.zeros((params['n'], params['d']), dtype=np.int32)
    Y = np.zeros(params['n'], dtype=np.int32)

    subpops = np.zeros(params['n'], dtype=np.int32)
    subpop_counts = np.zeros(params['N'], dtype=np.int32) # track how many points came from each subpop

    for i in range(params['n']):
        X[i,:], Y[i], subpops[i] = sample_uniform(instance, params)
        subpop_counts[Y[i]] += 1

    # training set of just singletons
    singleton_index = np.arange(params['N'])[subpop_counts == 1]
    train_singles_X = X[singleton_index]
    train_singles_Y = Y[singleton_index]

    # generate testing data, 500 data points for now
    test_n = 500
    test_X = np.zeros((test_n, params['d']), dtype=np.int32)
    test_Y = np.zeros(test_n, dtype=np.int32)
    test_subpops = np.zeros(test_n, dtype=np.int32)
    test_subpop_counts = np.zeros(params['N'], dtype=np.int32) # track how many points came from each subpop
    for i in range(test_n):
        test_X[i,:], test_Y[i], test_subpops[i] = sample_uniform(instance, params)
        test_subpop_counts[test_subpops[i]] += 1

    # generate "represented subpop" test dataset
    represented = np.arange(params['N'])[subpop_counts > 0]
    represented_X = np.test_X = np.zeros((test_n, params['d']), dtype=np.int32)
    represented_Y = np.zeros(test_n, dtype=np.int32)
    represented_subpops = np.zeros(test_n, dtype=np.int32)
    represented_subpop_counts = np.zeros(params['N'], dtype=np.int32) # track how many points came from each subpop
    for i in range(test_n):
        subpop = np.random.choice(represented)
        represented_X[i,:], represented_Y[i], represented_subpops[i] = sample_from_subpop(instance, params, subpop)
        represented_subpop_counts[represented_subpops[i]] += 1

    # generate "singletons-only" test dataset
    singleton_index = np.arange(params['N'])[subpop_counts == 1]

    singles_X = np.test_X = np.zeros((test_n, params['d']), dtype=np.int32)
    singles_Y = np.zeros(test_n, dtype=np.int32)
    singles_subpops = np.zeros(test_n, dtype=np.int32)
    singles_subpop_counts = np.zeros(params['N'], dtype=np.int32) # track how many points came from each subpop
    for i in range(test_n):
        subpop = np.random.choice(singleton_index)
        singles_X[i,:], singles_Y[i], singles_subpops[i] = sample_from_subpop(instance, params, subpop)
        singles_subpop_counts[singles_subpops[i]] += 1

    X = torch.tensor(X, dtype=torch.float)
    Y = torch.tensor(Y, dtype=torch.long)
    train_singles_X = torch.tensor(train_singles_X, dtype=torch.float)
    train_singles_Y = torch.tensor(train_singles_Y, dtype=torch.long)
    test_X = torch.tensor(test_X, dtype=torch.float)
    test_Y = torch.tensor(test_Y, dtype=torch.long)
    represented_X = torch.tensor(represented_X, dtype=torch.float)
    represented_Y = torch.tensor(represented_Y, dtype=torch.long)
    singles_X = torch.tensor(singles_X, dtype=torch.float)
    singles_Y = torch.tensor(singles_Y, dtype=torch.long)

    return X, Y, train_singles_X, train_singles_Y, test_X, test_Y, represented_X, represented_Y, singles_X, singles_Y

def empirical_error(X, Y, model):
    """check error of model on X,y. Need model.predict() method"""
    try:
        torch_flag = model.torch_flag  # this is only defined for torch models
    except AttributeError:
        torch_flag = False
    if torch_flag:
        Y_hat = torch.argmax(model(X), axis=1).numpy()
    else:
        Y_hat = model.predict(X)
    return np.count_nonzero(Y_hat==Y.numpy()) / X.shape[0]

def calculate_various_errors(model, data_tuple, test='all', verbose=True):
    """
    Calculate errors on different test datasets.
    'test' is list with which ones to calculate, or string 'all'
    """
    train_acc = train_singles_acc = test_acc = represented_acc = singles_acc = None
    if test == 'all':
        test = ['train', 'train_singles', 'test', 'represented', 'singles']
    if 'train' in test:
        train_acc = empirical_error(data_tuple[0], data_tuple[1], model)
        if verbose:
            print('Training set accuracy:         ', train_acc)
    if 'train_singles' in test:
        train_singles_acc = empirical_error(data_tuple[2], data_tuple[3], model)
        if verbose:
            print('Training set singletons accuracy:         ', train_acc)
    if 'test' in test:
        test_acc = empirical_error(data_tuple[4], data_tuple[5], model)
        if verbose:
            print('Fresh i.i.d. data accuracy:    ', test_acc)
    if 'represented' in test:
        represented_acc = empirical_error(data_tuple[6], data_tuple[7], model)
        if verbose:
            print('Fresh from subpops in the data:', represented_acc)
    if 'singles' in test:
        singles_acc = empirical_error(data_tuple[8], data_tuple[9], model)
        if verbose:
            print('Fresh from singleton subpops:  ', singles_acc)
    return train_acc, train_singles_acc, test_acc, represented_acc, singles_acc

def find_singletons(Y):
    n = len(Y)
    subpop_counts = np.zeros(n, dtype=np.int32)
    for i in range(n):
        subpop_counts[Y[i]] += 1
    singletons = []
    for i in range(n):
        if subpop_counts[Y[i]] == 1:
            singletons.append(i)
    return singletons
    



