import re
import os
import numpy as np
from featuresharing.visualization import multiplot_accuracies

datadir = 'data/processed'
groups = [
    [
        'alexnet-large-objectsvstextinthewild/Object+Text/raw/random-20/',
        'alexnet-large-objectsvstextinthewild/Object+Text/raw/x/',
        'alexnet-large-objectsvstextinthewild/Object+Text/raw/y/',
    ],
    [
        'alexnet-large-basicvssubord/Subord+Basic/raw/random-20/',
        'alexnet-large-basicvssubord/Subord+Basic/raw/x/',
        'alexnet-large-basicvssubord/Subord+Basic/raw/y/',
    ]
]
outputfile = [
    'data/processed/plot_cumlesioning_text_subord_random+x+y.png',
    'data/processed/plot_cumlesioning_basic_subord_random+x+y.png',
]
labels = [
    ['Object recognition', 'Text recognition'],
    ['Subordinate level rec', 'Basic level recognition'],
]

for i in range(len(groups)):
    accuracies = []
    layers = []

    for experiment in groups[i]:
        expdir = os.path.join(datadir, experiment)
        files = [os.path.join(expdir, f) for f in os.listdir(expdir) if f.startswith('data-accuracy')]
        files.sort()

        accs = []
        for file in files:
            accs.append(np.load(file))
        accuracies.append(accs)

        layers.append([re.search(".*_(.*)\.npy", f).group(1) for f in files])

    print('Saving plot to %s' % outputfile[i])
    multiplot_accuracies(outputfile[i], accuracies, layers, labels[i])
