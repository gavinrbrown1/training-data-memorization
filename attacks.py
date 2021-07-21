# Code for experiments in
# "When Is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?"
# Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar

# This file contains functions to attack models

import numpy as np
import torch
import math    

from base_utils import sample_from_subpop

def score(model, x, goal_label):
    """wrap the prediction, depending on if its torch or not"""
    try:
        torch_flag = model.torch_flag
    except AttributeError:
        torch_flag = False
    #torch_flag = True ############# delete me!!!#####
    if torch_flag:
        return model(x)[0,goal_label].item()
    else:
        return model.predict_proba(x.reshape(1,-1))[0,goal_label]

def coordinate_ascent(model, start_x='random', goal_label=0, params=None,
                      max_epoch=3, selection_method='ordered', verbose=False):
    """
    Return input that approximately maximizes probability of goal.
    Use coordinate-ascent.
    model() must return some maximizable values
    """
    if selection_method not in ['ordered', 'randompermutation', 'greedy']:
        print('Need a valid selection_method')
        return

    score_history = []   # track the scores over runs
    if start_x == 'random':
        start_x = np.random.choice([-1,+1], size=params['d'])
        start_x = torch.reshape(torch.tensor(start_x, dtype=torch.float), (1,params['d']))

    current_x = start_x.detach().clone()
    current_score = score(model, current_x, goal_label)
    if verbose:
        print(current_x)
        print(current_score)
        print()
    score_history.append(current_score)
    updated_this_epoch = True  # flag to check for convergence
    for epoch in range(max_epoch):
        if not updated_this_epoch:
            #print('Coordinate ascent converged')
            return current_x, score_history
        else:
            updated_this_epoch = False

        # set up order for indices, if not using greedy. If greedy, epoch_permutation is unused
        if (selection_method == 'ordered') and (epoch == 0):  # only need to set this the first time
            epoch_permutation = np.arange(params['d'])
        elif selection_method == 'randompermutation':
            epoch_permutation = np.random.permutation(params['d'])

        # iterate over indices, proposing new value
        for t in range(params['d']):  # one epoch consists of d proposals
            # select next index
            if selection_method != 'greedy':
                index = epoch_permutation[t]
            else:
                best_i = None
                best_score = current_score
                proposal_proposal_x = current_x.detach().clone()
                for i in range(params['d']):
                    proposal_proposal_x[0,i] = -1 * proposal_proposal_x[0,i]  # flip the bit for proposal
                    proposal_proposal_score = score(model, proposal_proposal_x, goal_label)
                    proposal_proposal_x[0,i] = -1 * proposal_proposal_x[0,i]  # flip the bit back
                    if proposal_proposal_score > best_score:
                        best_score = proposal_proposal_score
                        best_i = i
                if best_i is None:  # then we are at a local maximum
                    print('Coordinate ascent converged')
                    return current_x, score_history
                else:
                    index = best_i # can update current_x here, but do it below for harmony with nongreedy

            # accept or reject
            proposal_x = current_x.detach().clone()
            proposal_x[0,index] = -1 * proposal_x[0,index]  # flip the bit for proposal
            proposal_score = score(model, proposal_x, goal_label)
            if verbose:
                print('proposal:')
                print(proposal_x)
                print(proposal_score)
            if proposal_score > current_score:
                if verbose:
                    print('accepted', end='\n\n')
                updated_this_epoch = True
                current_score = proposal_score
                current_x = proposal_x
                score_history.append(current_score)
            else:
                if verbose:
                    print('rejected', end='\n\n')
    print('max_epoch reached, coordinate ascent attack did not converge.')
    return current_x, score_history

def sign_attack(model, adversary_type='informed', goal_label=0, instance=None, params=None, repeats=3, verbose=False):
    """
    Return input that approximately maximizes probability of goal.
    Use a modification of a 'gradient sign' attack.
    model() must return some maximizable values
    """
    if adversary_type not in ['informed', 'uninformed']:
        print('Need a valid adversary_type')
        return

    results = np.zeros(shape=(repeats, params['d']), dtype=np.int32)
    for k in range(repeats):
        for i in range(params['d']):
            # set up the point to test from
            if adversary_type == 'uninformed':
                x = np.random.choice([-1,+1], params['d'])
            else:
                if instance is None:
                    print('An informed adversary needs to know the instance!')
                    return
                x, y, subpop = sample_from_subpop(instance, params, goal_label)
            # try both points now
            x = torch.reshape(torch.tensor(x, dtype=torch.float), (1,params['d']))
            x[0,i] = +1
            up_score = score(model, x, goal_label)
            x[0,i] = -1
            down_score = score(model, x, goal_label)
            if up_score > down_score:
                results[k,i] = +1
            else:
                results[k,i] = -1
    # just take the sum along "repeats" axis, which will be positive or negative
    out = np.sum(results, axis=0)
    out[out > 0] = +1
    out[out < 0] = -1
    return torch.tensor(out)


                






