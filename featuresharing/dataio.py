import os
import random
import numpy as np


def read(imagesrc, imagelist):
    data = []
    targets = []
    with open(imagelist, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            sep = line.rfind(' ')
            filep, categories = line[:sep], line[sep:]
            data.append(os.path.join(imagesrc, filep))
            targets.append([int(c) for c in categories.split(',')])

    shuffle_inds = range(len(data))
    random.seed(0)
    random.shuffle(shuffle_inds)
    data = [data[i] for i in shuffle_inds]
    targets = [targets[i] for i in shuffle_inds]

    return data, targets


def loadnpz(f):
    dat = np.load(f)
    return [dat[key] for key in dat], [key for key in dat]