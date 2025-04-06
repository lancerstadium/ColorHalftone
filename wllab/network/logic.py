import numpy as np
import torch
import torch.nn as nn
from ..common.logic import *


def get_logic_model(
        grad_factor=1.1, 
        connections='unique', 
        tau=10, 
        architecture='randomly_connected', 
        num_neurons=1200, 
        num_layers=6, 
        layer_neurons=None,
        learning_rate=0.01, 
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
        if layer_neurons is None:
            logic_layers.append(torch.nn.Flatten())
            logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=k, **llkw))
            for _ in range(l - 1):
                logic_layers.append(LogicLayer(in_dim=k, out_dim=k, **llkw))

            model = torch.nn.Sequential(
                *logic_layers,
                GroupSum(class_count, tau)
            )
        else:
            nl = len(layer_neurons)
            logic_layers.append(torch.nn.Flatten())
            logic_layers.append(LogicLayer(in_dim=in_dim, out_dim=layer_neurons[0], **llkw))
            for i in range(1, nl):
                logic_layers.append(LogicLayer(in_dim=layer_neurons[i-1], out_dim=layer_neurons[i], **llkw))

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


