import sys
import torch
import dataio
import analysis
import preprocessing
import visualization
import numpy as np
from collections import OrderedDict
from torch.utils.serialization import load_lua
from nnadapter.torchadapter import TorchAdapter


def init_model(args):
    # load class to indices
    lua_gen = load_lua(args.classtoidx)
    classids_to_pred = [dict(d) for d in lua_gen[0]['train']['classToIdx']]

    # load model via adapter
    nn = TorchAdapter(args.model,
                      mean=np.array([0.485, 0.456, 0.406]),
                      std=np.array([0.229, 0.224, 0.225]),
                      inputsize=[3, 224, 224],
                      use_gpu=True,
                      output_filter=[str(i) for i in range(20)]
                      )
    nn.model = torch.nn.DataParallel(nn.model)

    return nn, classids_to_pred


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--images', type=str, required=True, help='Path to images')
    parser.add_argument('--imagelist', type=str, required=True, help='Path to caffe style image list')
    parser.add_argument('--categories', type=str, required=True, nargs=2, help='Categories')
    parser.add_argument('--classtoidx', type=str, required=True,
                        help='Path to torch7 .t7 file containing the class to idx dictionary.')
    parser.add_argument('--output', type=str, required=True, help='Output for data, logs and plots')
    parser.add_argument('--layers', type=str, required=True, nargs='+', help='Layers to probe')
    parser.add_argument('--softmax', type=str, required=False, default='True',
                        help='Whether to probe softmax or raw output')
    parser.add_argument('--scoremethod', type=str, required=False, default='x-y', help='Legal values: x-y, y-x, x, y')
    parser.add_argument('--random', type=int, default=0, help='Evaluate n random unit lesion sequence, if n > 0.')
    parser.add_argument('--visualize', type=bool, required=False, default=False, help='Visualize features')

    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate arguments and create output directories

    args.softmax = args.softmax == 'True'
    FIG_OUT = os.path.join(args.output,
                           '+'.join(args.categories),
                           'softmax' if args.softmax else 'raw',
                           'random-{}'.format(args.random) if args.random > 0 else args.scoremethod)
    if not os.path.exists(FIG_OUT):
        os.makedirs(FIG_OUT)
    print('Saving to: %s' % FIG_OUT)

    layer_names = {'0.0.0': 'conv1',
                   '0.0.3': 'conv2',
                   '0.0.6': 'conv3',
                   '0.0.8': 'conv4',
                   '0.0.10': 'conv5',
                   '1.2': 'fc6',
                   '1.5': 'fc7',
                   '1.7.0': 'fc8-task1',
                   '1.7.1': 'fc8-task2'}

    for layer in args.layers:
        if layer not in layer_names:
            raise KeyError('Layer %s is not known. Registered layers are: %s' % (layer, layer_names.keys()))

    scoring_methods = {
        'x-y': lambda x, y: x - y,
        'y-x': lambda x, y: y - x,
        'x': lambda x, y: x,
        'y': lambda x, y: y,
    }
    scoring = scoring_methods[args.scoremethod]

    # ------------------------------------------------------------------------------------------------------------------
    # Load and initialize model
    nn, classids_to_pred = init_model(args)

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize features
    if args.visualize:
        feature_output_folder = os.path.join(args.output, '+'.join(args.categories), 'deepvis')
        if not os.path.exists(feature_output_folder):
            os.makedirs(feature_output_folder)

        print('Saving visualizations to: %s' % feature_output_folder)
        visualization.visualize_features(args, nn, layer_names)

        sys.exit(0)

    # ------------------------------------------------------------------------------------------------------------------
    # Construct list of images and corresponding targets from args.imagelist
    data, targetlists = dataio.read(args.images, args.imagelist)
    # convert target classes to indices the model understands
    for i in range(len(targetlists)):
        for task in range(len(targetlists[i])):
            targetlists[i][task] = classids_to_pred[task][targetlists[i][task]] - 1
    targets = np.array(targetlists)

    # ------------------------------------------------------------------------------------------------------------------
    # Remove images that are incorrectly classified
    batch = nn.preprocess(data)
    predictions = preprocessing.classify(nn, batch, 256, progressbar=True)
    correct_preds = preprocessing.prediction_equals(predictions, targets)
    correct_indices = np.sum(correct_preds, axis=0) == 2

    batch = batch[correct_indices]
    targets = targets[correct_indices]

    # limit batch to 600 elements, to make it fit into 12GB gpu
    batch = batch[:600]
    targets = targets[:600]
    batchsize = len(batch)

    print('Remaining samples after filter: %d' % batchsize)

    # ------------------------------------------------------------------------------------------------------------------
    # Volatile lesioning to determine each features impact to target output
    if args.random <= 0:
        print('Volatile lesioning...')
        importance_ordered_indices = analysis.volatile_lesioning(args, nn, batch, targets, batchsize, scoring)
    else:
        importance_ordered_indices = OrderedDict()
        for layer in args.layers:
            (weights, bias) = nn.get_layerparams(layer)
            importance_ordered_indices[layer] = np.arange(weights.shape[0])

    # ------------------------------------------------------------------------------------------------------------------
    # Persistent lesioning to determine decay of classification accuracies
    print('Persistent lesioning...')
    accuracies_by_layer = analysis.persistent_lesioning(args, nn, batch, targets, importance_ordered_indices)

    # ------------------------------------------------------------------------------------------------------------------
    # Save data
    for layer, accuracies in accuracies_by_layer.items():
        name = layer_names[layer]
        np.save(os.path.join(FIG_OUT, 'data-accuracy-change_layer_%s.npy' % name), accuracies)
        np.save(os.path.join(FIG_OUT, 'data-ordered-indices_layer_%s.npy' % name), importance_ordered_indices)

    # ------------------------------------------------------------------------------------------------------------------
    # Visualize accuracy decays
    for layer, accuracies in accuracies_by_layer.items():
        filep = os.path.join(FIG_OUT, 'accuracy-change_layer_%s.png' % layer_names[layer])
        visualization.plot_accuracies(filep,
                                      accuracies,
                                      ['%s recognition' % args.categories[0], '%s recognition' % args.categories[1]],
                                      'Layer %s' % layer_names[layer],
                                      'Top-1 Accuracy',
                                      'Number of units removed'
                                      )



