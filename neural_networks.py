# Code for experiments in
# "When Is Memorization of Irrelevant Training Data Necessary for High-Accuracy Learning?"
# Gavin Brown, Mark Bun, Vitaly Feldman, Adam Smith, and Kunal Talwar

# NN classes (including logistic regression)

import torch
from base_utils import calculate_various_errors

class TorchLogit(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(TorchLogit, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, D_out)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        self.torch_flag = True  # used to contrast with non-torch methods

    def forward(self, x):
        scores = self.linear1(x)
        y_pred = self.logsoftmax(scores)
        if False:
            print(x.shape)
            print(x)
            print(scores.shape)
            print(scores)
            print(torch.max(torch.sum(y_pred, dim=1)))
            print(torch.min(torch.sum(y_pred, dim=1)))
            print(y_pred.shape)
            print(y_pred)
        return y_pred

class TorchMLP(torch.nn.Module):
    def __init__(self, D_in, D_out, hidden_layers, activation, dropout_rate):
        super(TorchMLP, self).__init__()
        self.l = len(hidden_layers)
        if self.l > 3:
            raise ValueError('Currently only accept <= 3 hidden layers, b/c ModuleList wasn\'t working.')
        # this part could be shortened, hopefully the more-explicit version is more-checkable
        if self.l == 1:
            self.linear1 = torch.nn.Linear(D_in, hidden_layers[0])
            self.linear2 = torch.nn.Linear(hidden_layers[0], D_out)
        elif self.l == 2:
            self.linear1 = torch.nn.Linear(D_in, hidden_layers[0])
            self.linear2 = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
            self.linear3 = torch.nn.Linear(hidden_layers[1], D_out)
        else:
            self.linear1 = torch.nn.Linear(D_in, hidden_layers[0])
            self.linear2 = torch.nn.Linear(hidden_layers[0], hidden_layers[1])
            self.linear3 = torch.nn.Linear(hidden_layers[1], hidden_layers[2])
            self.linear4 = torch.nn.Linear(hidden_layers[2], D_out)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)
        if activation == 'relu':
            self.activation = torch.nn.ReLU()
        else:
            self.activation = torch.nn.Sigmoid()
        self.torch_flag = True

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        if self.l > 1:
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear3(x)
            if self.l == 3:
                x = self.activation(x)
                x = self.dropout(x)
                x = self.linear4(x)
        return self.logsoftmax(x)
    

