import numpy as np
import torch
import torch.nn as nn
from ..common.logic import *


def get_logic_model(
        grad_factor, 
        connections, 
        tau, 
        architecture, 
        num_neurons, 
        num_layers, 
        learning_rate, 
        input_ndim=784, 
        nclasses=10
    ):
    llkw = dict(grad_factor=grad_factor, connections=connections)

    in_dim = input_ndim
    class_count = nclasses

    logic_layers = []

    arch = architecture
    k = num_neurons
    l = num_layers

    ####################################################################################################################

    if arch == 'randomly_connected':
        logic_layers.append(torch.nn.Flatten())
        logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
        for _ in range(l - 1):
            logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

        model = torch.nn.Sequential(
            *logic_layers,
            GroupSum(class_count, tau)
        )

    ####################################################################################################################

    else:
        raise NotImplementedError(arch)

    ####################################################################################################################

    total_num_neurons = sum(map(lambda x: x.num_neurons, logic_layers[1:-1]))
    print(f'total_num_neurons={total_num_neurons}')
    total_num_weights = sum(map(lambda x: x.num_weights, logic_layers[1:-1]))
    print(f'total_num_weights={total_num_weights}')

    model = model.to('cuda')
    print(model)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    return model, loss_fn, optimizer