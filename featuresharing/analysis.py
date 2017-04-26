import torch
import numpy as np
import preprocessing
from tqdm import tqdm
from collections import OrderedDict


def softmax(x):
    e_x = np.exp(x - x.max(axis=1)[:, None])
    sm = e_x / e_x.sum(axis=1)[:, None]
    return sm


def probe_unit(nn, weights, bias, layer, batch, unitidx, original_preds, targets, apply_softmax=False):
    weight_backup = weights[unitidx].copy()
    bias_backup = bias[unitidx].copy()

    weights[unitidx] = 0
    bias[unitidx] = 0
    nn.set_weights(layer, weights)
    nn.set_bias(layer, bias)

    preds = nn.forward(batch)

    if apply_softmax:
        # apply softmax to predictions
        for task in range(len(preds)):
            preds[task] = softmax(preds[task])

    diffs = []

    for task in range(len(preds)):
        inds_axis0 = np.arange(targets.shape[0])
        inds_axis1 = targets[:, task]
        taskpreddiff = preds[task][inds_axis0, inds_axis1] - original_preds[task][inds_axis0, inds_axis1]
        diffs.append(taskpreddiff)

    weights[unitidx] = weight_backup
    bias[unitidx] = bias_backup

    return diffs


def volatile_lesioning(args, nn, batch, targets, batchsize, scoringfunc):
    lesion_impacts = OrderedDict()
    importance_ordered_indices = OrderedDict()

    for i in range(0, len(batch), batchsize):
        minibatch = batch[i:(i + batchsize)]
        targetbatch = targets[i:(i + batchsize)]

        preds = nn.forward(minibatch)

        for layer in args.layers:
            (weights, bias) = nn.get_layerparams(layer)

            lesion_impact = np.zeros((weights.shape[0], 2, minibatch.shape[0]))

            for unitidx in tqdm(range(weights.shape[0]), desc='Layer '+layer):
                diffs = probe_unit(nn, weights, bias, layer, batch, unitidx, preds, targetbatch, args.softmax)
                lesion_impact[unitidx, :, :] = diffs

            if layer not in lesion_impacts:
                lesion_impacts[layer] = lesion_impact
            else:
                lesion_impacts[layer] = np.concatenate([lesion_impacts[layer], lesion_impact], axis=2)

            # ensure original sets of weights and bias are restored
            nn.set_weights(layer, weights)
            nn.set_bias(layer, bias)

    for layer, lesion_impact in lesion_impacts.items():
        lesion_impact = lesion_impact.mean(2)

        diff = scoringfunc(lesion_impact[:, 0], lesion_impact[:, 1])
        importance_ordered_indices[layer] = np.argsort(diff)

    return importance_ordered_indices


def persistent_lesioning(args, nn, batch, targets, lesion_order):
    nsamples = max(1, args.random)
    accuracies_by_layer = OrderedDict()

    for layer, ind_order in lesion_order.items():

        weights, bias = nn.get_layerparams(layer)
        weights_backup, bias_backup = weights.copy(), bias.copy()

        accuracies = np.zeros((nsamples, len(ind_order) + 1, targets.shape[1]))

        for run in range(nsamples):
            if args.random > 0:
                inds = np.random.permutation(ind_order)
            else:
                inds = ind_order

            preds = nn.forward(batch)
            accuracies[run, 0, :] = preprocessing.accuracy(preds, targets)

            for k, ind in enumerate(tqdm(inds, desc='Layer ' + layer)):
                weights[ind] = 0
                bias[ind] = 0
                nn.set_weights(layer, weights)
                nn.set_bias(layer, bias)

                preds = nn.forward(batch)

                accuracies[run, k + 1, :] = preprocessing.accuracy(preds, targets)

            # ensure original sets of weights and bias are restored
            nn.set_weights(layer, weights_backup)
            nn.set_bias(layer, bias_backup)

        accuracies_by_layer[layer] = accuracies

    return accuracies_by_layer
