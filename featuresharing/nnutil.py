import numpy as np
from tqdm import tqdm
from torch.utils.serialization import load_lua
from emu.torch import TorchAdapter


def batchify(x, dim, size):
    assert size > 0

    length = x.shape[dim]

    for i in range(0, length, size):
        yield x.take(range(i, min(length, i + size)), axis=dim)


def softmax(x):
    e_x = np.exp(x - x.max(axis=1)[:, None])
    sm = e_x / e_x.sum(axis=1)[:, None]
    return sm


def log_softmax(x):
    xdev = x - x.max(axis=1)[:, None]
    lsm = xdev - np.log(np.sum(np.exp(xdev), axis=1, keepdims=True))
    return lsm


def init_model(model_fp, classtoidx_fp):
    # load class to indices
    lua_gen = load_lua(classtoidx_fp)
    classids_to_pred = [dict(d) for d in lua_gen[0]['train']['classToIdx']]

    # load model via adapter
    nn = TorchAdapter(model_fp,
                      mean=np.array([0.485, 0.456, 0.406]),
                      std=np.array([0.229, 0.224, 0.225]),
                      inputsize=[3, 224, 224],
                      use_gpu=True,
                      )

    return nn, classids_to_pred


def classify(nn, data, batchsize, apply_softmax=False, progressbar=False):
    """
    Expects preprocessed data
    """
    preds = None

    pbar = (lambda x, **kwargs: x) if not progressbar else tqdm

    for i in pbar(range(0, data.shape[0], batchsize), desc='Classification'):
        minibatch = data[i:(i + batchsize)]
        out = [softmax(x) if apply_softmax else x for x in nn.forward(minibatch)]

        if preds is None:
            preds = []
            for pred in out:
                preds.append(np.empty((data.shape[0], pred.shape[1])))

        for task in range(len(preds)):
            preds[task][i:(i + batchsize), :] = out[task]

    return preds


def prediction_equals(predictions, targets):
    correct = []
    for task in range(len(predictions)):
        inds = np.argmax(predictions[task], axis=1)
        correct.append(inds == targets[:, task])

    return np.array(correct)


def accuracy(predictions, targets):
    return np.sum(prediction_equals(predictions, targets), axis=1) / float(len(targets))