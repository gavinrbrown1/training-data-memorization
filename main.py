# Code for experiments in
# "When Is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?"
# Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar

# main script: generate a data set, train and attack models, and produce figures 
# runs under 10 minutes on a macbook pro

import numpy as np
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from base_utils import *
from neural_networks import TorchMLP, TorchLogit, train_logit
from attacks import sign_attack, coordinate_ascent

tic = time()

# set params, make data 
params = {'n': 500, 'seed': 123}
params['d'] = 2 * params['n']
params['N'] = params['n']
a = 50000
params['rho'] = math.sqrt((2 * math.log(a * 0.368 * params['n']) - math.log(math.log(params['n']))) / params['d'])

print('rho:', params['rho'])

# set the seeds before generating data
np.random.seed(params['seed'])

# generate a problem instance and a data set
print('Generating data')
instance = create_problem_instance(params)
data_tuple = make_data(params, instance)
X, Y, train_singles_X, train_singles_Y, test_X, test_Y, represented_X, represented_Y, singles_X, singles_Y = data_tuple

# upsample the singletons
singletons = find_singletons(Y)
additional_singletons = torch.zeros((len(singletons),params['d']), dtype=torch.float)
additional_labels = torch.zeros(len(singletons), dtype=torch.long)
for i, singleton in enumerate(singletons):
    additional_singletons[i] = X[singleton,:]
    additional_labels[i] = Y[singleton]

upsampled_X = torch.vstack((X,additional_singletons))
upsampled_Y = torch.hstack((Y,additional_labels))

# logistic regression
print('training logit')
epochs = 50 
results = pd.DataFrame(columns=['epoch','train_loss','train_acc','train_sing_acc','test_acc','repr_acc','sing_acc','rec_acc_grad', 'rec_acc_coord'])
epoch_results = {'epoch':-1, 'train_loss':-1, 'train_acc':-1, 'train_sing_acc':-1,'test_acc':-1, 'repr_acc':-1, 'sing_acc':-1, 'rec_acc_grad':-1, 'rec_acc_coord':-1}

model = TorchLogit(params['d'],params['N'])
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.99, weight_decay=0, nesterov=True)
for t in range(epochs):
    Y_hat = model(upsampled_X)

    loss = loss_fn(Y_hat, upsampled_Y)
    if t % 2 == 0:
        print('gradient step '+str(t)+' out of 50')
        with torch.no_grad():
            # calculate some accuracies
            accs = calculate_various_errors(model, data_tuple, test='all', verbose=False)
            epoch_results['epoch'] = t
            epoch_results['train_loss'] = loss.item()
            epoch_results['train_acc'] = accs[0]
            epoch_results['train_sing_acc'] = accs[1]
            epoch_results['test_acc'] = accs[2]
            epoch_results['repr_acc'] = accs[3]
            epoch_results['sing_acc'] = accs[4]

            # do some attacks
            num_attack = 3 
            missed_bits = np.zeros(num_attack)
            for i, singleton in enumerate(singletons[:num_attack]):
                out = sign_attack(model, adversary_type='uninformed', goal_label=Y[singleton], instance=None, params=params, repeats=5, verbose=False)
                missed_bits[i] = np.count_nonzero(X[singleton,:].numpy() != out.numpy())
            epoch_results['rec_acc_grad'] = 1 - (np.mean(missed_bits)/params['d'])

        results = results.append(epoch_results, ignore_index=True)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
model.eval()

results.to_csv('logit_training.csv')

# mlp
results = pd.DataFrame(columns=['epoch','train_loss','train_sing_acc','test_acc','sing_acc','rec_acc_grad','rec_acc_coord'])
epoch_results = {'epoch':-1, 'train_loss':-1, 'train_sing_acc':-1,'test_acc':-1, 'sing_acc':-1, 'rec_acc_grad':-1, 'rec_acc_coord':-1}

# set random seeds, for the execution of training and attack
np.random.seed(params['seed'])
torch.manual_seed(params['seed'])

## initialize the model
model = TorchMLP(params['d'],params['N'],(1500,),activation='sigmoid', dropout_rate=0)

loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.999, weight_decay=0.0, nesterov=True)

lambda1 = lambda epoch: 1 if (epoch < 450) else 3e-1  # fn to multiply with base lr_rate
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda1) 

## train away!
print('training mlp.')
for t in range(2000):
    Y_hat = model(X)
    loss = loss_fn(Y_hat, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    if t % 50 == 0:
        print('gradient step '+str(t)+' out of 2000')
        with torch.no_grad():
            # calculate some accuracies
            accs = calculate_various_errors(model, data_tuple, test='all', verbose=False)
            epoch_results['epoch'] = t
            epoch_results['train_loss'] = loss.item()
            epoch_results['train_acc'] = accs[0]
            epoch_results['train_sing_acc'] = accs[1]
            epoch_results['test_acc'] = accs[2]
            epoch_results['repr_acc'] = accs[3]
            epoch_results['sing_acc'] = accs[4]
            
            # coordinate ascent attack
            num_attack = 3 
            missed_bits = np.zeros(num_attack)
            for i, singleton in enumerate(singletons[:num_attack]):
                out, score_history = coordinate_ascent(model, start_x='random', goal_label=Y[singleton], 
                                                        params=params, max_epoch=10, selection_method='ordered',verbose=False)
                missed_bits[i] = np.count_nonzero(X[singleton,:].numpy() != out.numpy())
            epoch_results['rec_acc_coord'] = 1 - (np.mean(missed_bits)/params['d'])

        results = results.append(epoch_results, ignore_index=True)
model.eval()

results.to_csv('mlp_training.csv')

# make plots

fontsize = 15 
linewidth = 3
plt.rc('font', size=fontsize)
plt.rc('axes', labelsize=fontsize)
plt.rc('xtick', labelsize=fontsize-2)
plt.rc('ytick', labelsize=fontsize-2)
plt.rc('legend', fontsize=fontsize-2)
plt.rc('figure', titlesize=fontsize)
plt.rc('lines', linewidth=linewidth)

# make logit plot
results = pd.read_csv('logit_training.csv')

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Gradient Updates')
ax1.set_ylabel('Recovery Error', color=color)
p1, = ax1.plot(results['epoch'], 1 - results['rec_acc_grad'], color=color, label='Attack Recovery Error')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Classification Error', color=color)
p2, = ax2.plot(results['epoch'], 1 - results['train_acc'], '--', color=color, label='Train Set Classification Error')
p3, = ax2.plot(results['epoch'], 1 - results['repr_acc'], '-.', color=color, label='Represented Classification Error')
p4, = ax2.plot(results['epoch'], 1 - results['sing_acc'], ':', color=color, label='Singletons Classification Error')
ax2.tick_params(axis='y', labelcolor=color)

plt.legend(handles=[p1, p2, p3, p4])
plt.title('Logistic Regression, Gradient Attack')
fig.tight_layout()
plt.savefig('logit_figure.png')

# make mlp plot
results = pd.read_csv('mlp_training.csv')

fig, ax1 = plt.subplots()
fig.subplots_adjust(right=0.75)

color = 'tab:red'
ax1.set_xlabel('Gradient Updates')
ax1.set_ylabel('Recovery Error', color=color)
p1, = ax1.plot(results['epoch'], 1 - results['rec_acc_coord'], color=color, label='Coordinate Recovery Error')
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:blue'
ax2.set_ylabel('Classification Error', color=color)
p2, = ax2.plot(results['epoch'], 1 - results['train_acc'], '--', color=color, label='Train Set (Classification Error)')
p3, = ax2.plot(results['epoch'], 1 - results['repr_acc'], '-.', color=color, label='Represented (Classification Error)')
p4, = ax2.plot(results['epoch'], 1 - results['sing_acc'], ':', color=color, label='Represented (Classification Error)')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Multilayer Perceptron, Coordinate Attack')
fig.tight_layout()
plt.savefig('mlp_figure.png')

toc = time()
print('total time:', toc-tic)
