import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.autograd import Variable
import json

from ml import nets
import numpy as np


def evaluate(model_suffix, dataloader, parameters,
             models_directory='../models/'):
    batch_size = parameters.get('batch_size', 100)
    hidden_sizes = parameters.get('hidden_sizes', 200)
    input_dim = parameters.get('input_dim')
    output_dim = parameters.get('output_dim')
    model_prefix = parameters.get('model_prefix', None)
    use_cuda = parameters.get('use_cuda')

    criterion = nn.CrossEntropyLoss()
    net = nets.AdvancedNet(input_dim, hidden_sizes, output_dim)
    if use_cuda:
        criterion = criterion.cuda()
        net = net.cuda()
    net.load_state_dict(
        torch.load(
            models_directory +
            model_prefix +
            '.' +
            model_suffix))

    accuracy, total_loss = eval_error(net, dataloader, criterion, use_cuda)
    print('Test acc: %.4f, loss: %.4f' % (accuracy, total_loss))


def eval_error(net, loader, criterion=None, use_cuda=True):
    num_batches = len(loader)
    total_loss = 0.0
    accuracy = 0.0
    net.eval()
    for inputs, labels in loader:
        if use_cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        net = net.double()  # to fix error: RuntimeError: expected scalar type Float but found Double
        outputs = net(inputs)
        if criterion is not None:
            total_loss += criterion(outputs, labels).item()
        max_index = outputs.max(dim=1)[1]
        accuracy += np.sum(max_index.data.cpu().numpy() ==
                           labels.data.cpu().numpy()) / inputs.size()[0]

    net.train()
    accuracy = accuracy / num_batches
    total_loss = total_loss / num_batches
    return accuracy, total_loss


# print predictions of the NN, see paper_ground_truth.py
def print_predictions(X, fids, parameters, mapping,
                      file_name, models_directory='../models/'):
    hidden_sizes = parameters.get('hidden_sizes', 200)
    output_dim = parameters.get('output_dim', -1)
    model_prefix = parameters.get('model_prefix', None)
    use_cuda = parameters.get('use_cuda')

    net = nets.AdvancedNet(X.shape[1], hidden_sizes, output_dim)
    net.load_state_dict(torch.load(models_directory + model_prefix))
    X = Variable(torch.from_numpy(X).float())
    if use_cuda:
        net = net.cuda()
        X = X.cuda()
    net.eval()
    y = net(X)
    max_index = y.max(dim=1)[1]

    output_dict = {}
    for i in range(len(fids)):
        output_dict[fids[i]] = mapping[max_index[i].item()]

    with open(file_name, 'w') as fp:
        json.dump(output_dict, fp, indent=4)
