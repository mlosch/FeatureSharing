import numpy as np
from tqdm import tqdm


def batchify(x, dim, size):
    assert size > 0

    length = x.shape[dim]

    for i in xrange(0, length, size):
        yield x.take(range(i, min(length, i + size)), axis=dim)


def classify(nn, data, batchsize, progressbar=False):
    """
    Expects preprocess data
    """
    preds = None

    pbar = (lambda x, **kwargs: x) if not progressbar else tqdm

    for minibatch in pbar(batchify(data, 0, batchsize), desc='Classification'):
        minipreds = nn.forward(minibatch)
        if preds is None:
            preds = minipreds
        else:
            for task in range(len(preds)):
                preds[task] = np.concatenate((preds[task], minipreds[task]), axis=0)

    return preds


def prediction_equals(predictions, targets):
    correct = []
    for task in xrange(len(predictions)):
        inds = np.argmax(predictions[task], axis=1)
        correct.append(inds == targets[:, task])

    return np.array(correct)


def accuracy(predictions, targets):
    return np.sum(prediction_equals(predictions, targets), axis=1) / float(len(targets))
