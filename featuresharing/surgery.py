import torch
import nnutil
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from nnutil import softmax, log_softmax


class JointPdf(object):
    _LOG_2PI = np.log(2. * np.pi)

    def __init__(self, weight_mean, bias_mean):
        self.wmu = weight_mean
        self.bmu = bias_mean

    def _log_normal_pdf(self, mean, x):
        rank = mean.size
        norm_const = rank * self._LOG_2PI
        xdev = x-mean
        return -0.5 * (norm_const + np.sum(xdev**2, axis=-1))

    def logpdf(self, weight, bias):
        return self._log_normal_pdf(self.wmu, weight) + self._log_normal_pdf(self.bmu, bias)

    def rvs(self, n=1):
        weight_samples = np.random.randn(n, self.wmu.size) + self.wmu
        bias_samples = np.random.randn(n, self.bmu.size) + self.bmu
        return weight_samples, bias_samples.squeeze()


def parameter_priors(nn, layer):
    weights, bias = nn.get_layerparams(layer)

    priors = []
    for unitidx in range(weights.shape[0]):
        priors.append(JointPdf(weights[unitidx].ravel(), bias[unitidx].ravel()))

    return priors


def marginalize_out_feature(nn, batch, batchsize, layer, prior_pdf, nsamples):
    """
        Marginalization via:
        p(y | \Theta \\ w_i) = sum_{s=1}^{nsamples} p(w_i = w_s) p(y | \Theta \leftarrow w_i = w_s)
    """
    weights, bias = nn.get_layerparams(layer)
    nunits = weights.shape[0]
    ninputs = batch.shape[0]

    conditionals = []

    # determine output layer dimensions
    preds = nn.forward(batch[0][np.newaxis, ...])

    for pred in preds:
        noutputs = pred.shape[1]
        conditionals.append(np.empty((nunits, ninputs, noutputs), dtype=np.float32))

    for unitidx in tqdm(range(nunits), desc='Layer ' + layer):

        unit_weight = weights[unitidx].copy()
        unit_bias = bias[unitidx].copy()

        weight_samples, bias_samples = prior_pdf[unitidx].rvs(nsamples)
        # determine prior probability of sampled weight
        priors = prior_pdf[unitidx].logpdf(weight_samples, bias_samples)

        # --------------------------------------------------------------------------------------------------------------
        # as we undersample the parameter distribution, we here make sure that the probabilities of our samples sum to 1
        #   -> each sample represents thus the integral of its interval/volume
        # to circumvent issues with numerical instability for very small probabilities,
        # first ensure the highest probability is log(0)
        priors -= priors.max()
        priors -= np.log(np.exp(priors).sum())

        for i in range(0, len(batch), batchsize):
            minibatch = batch[i:(i + batchsize)]

            conditional_samples = []
            for pred in preds:
                noutputs = pred.shape[1]
                conditional_samples.append(np.empty((nsamples, minibatch.shape[0], noutputs)))

            for s in range(nsamples):

                # set weight and bias to sample
                nn.set_weights(layer, weights)
                nn.set_bias(layer, bias)

                # and classify to get p(y | \Theta \leftarrow w_i = w_s)
                preds = [log_softmax(x) for x in nn.forward(minibatch)]

                for task in range(len(preds)):
                    # preds[task] \in (nBatches x nOutputs)
                    conditional_samples[task][s, :, :] = np.exp(priors[s] + preds[task])

            for task in range(len(preds)):
                conditionals[task][unitidx, i:(i + batchsize), :] = conditional_samples[task].sum(0)

        # restore original parameter values
        weights[unitidx] = unit_weight
        bias[unitidx] = unit_bias

    nn.set_weights(layer, weights)
    nn.set_bias(layer, bias)

    return conditionals


def lesion_unit(nn, weights, bias, layer, batch, unitidx, apply_softmax=False):
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

    weights[unitidx] = weight_backup
    bias[unitidx] = bias_backup

    return preds


def volatile_lesioning(nn, batch, batchsize, layer, softmax):

    (weights, bias) = nn.get_layerparams(layer)
    nunits = weights.shape[0]
    ninputs = batch.shape[0]

    # determine output layer dimensions
    preds = nn.forward(batch[0][np.newaxis, ...])

    lesioned_preds = []
    for pred in preds:
        noutputs = pred.shape[1]
        lesioned_preds.append(np.empty((nunits, ninputs, noutputs), dtype=np.float32))

    for unitidx in tqdm(range(nunits), desc='Layer ' + layer):
        for i in range(0, len(batch), batchsize):
            minibatch = batch[i:(i + batchsize)]

            preds = lesion_unit(nn, weights, bias, layer, minibatch, unitidx, softmax)
            for task in range(len(preds)):
                lesioned_preds[task][unitidx, i:(i+batchsize), :] = preds[task]

    # ensure original sets of weights and bias are restored
    nn.set_weights(layer, weights)
    nn.set_bias(layer, bias)

    return lesioned_preds


def accuracy_batched(nn, batch, batchsize, targets):
    preds = nnutil.classify(nn, batch, batchsize)
    return nnutil.accuracy(preds, targets)

