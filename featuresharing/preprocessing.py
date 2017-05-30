import numpy as np
from featuresharing import dataio
from featuresharing import nnutil
from featuresharing import surgery


if __name__ == '__main__':
    from argparse import ArgumentParser
    import os

    parser = ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to model')
    parser.add_argument('--images', type=str, required=True, help='Path to images')
    parser.add_argument('--imagelist', type=str, required=True, help='Path to caffe style image list')
    parser.add_argument('--categories', type=str, required=True, nargs=2, help='Categories')
    parser.add_argument('--classtoidx', type=str, required=True,
                        help='Path to torch7 .t7 file containing the class to idx dictionary')
    parser.add_argument('--output', type=str, required=True, help='Output for processed data')
    parser.add_argument('--layers', type=str, required=True, nargs='+', help='Layers to process')
    parser.add_argument('--method', type=str, required=False, default='marginalize',
                        help='Legal values: marginalize, lesion')
    parser.add_argument('--softmax', type=str, required=False, default='True',
                        help='Whether to probe softmax or raw output. True if method == marginalize')
    parser.add_argument('--samples', type=int, default=20, help='If method == marginalize: '
                                                                'Number of parameter samples per feature unit')
    parser.add_argument('--batchsize', type=int, default=256)

    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate arguments and create output directories

    if args.method.lower() not in ['marginalize', 'lesion']:
        parser.error('Illegal method %s' % args.method)

    if args.method.lower() == 'marginalize':
        args.softmax = True
    else:
        args.softmax = args.softmax == 'True'

    DATA_OUT = os.path.join(args.output,
                            '+'.join(args.categories),
                            args.method.lower(),
                            'softmax' if args.softmax else 'raw')

    if not os.path.exists(DATA_OUT):
        os.makedirs(DATA_OUT)
    print('Saving to: %s' % DATA_OUT)

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

    # ------------------------------------------------------------------------------------------------------------------
    # Load and initialize model
    nn, classids_to_pred = nnutil.init_model(args.model, args.classtoidx)

    # ------------------------------------------------------------------------------------------------------------------
    # Construct list of images and corresponding targets from args.imagelist
    data, targetlists = dataio.read(args.images, args.imagelist)
    # convert target classes to indices the model understands
    for i in range(len(targetlists)):
        for task in range(len(targetlists[i])):
            targetlists[i][task] = classids_to_pred[task][targetlists[i][task]] - 1

    targets = np.array(targetlists)
    batch = nn.preprocess(data)

    batchsize = args.batchsize

    # ------------------------------------------------------------------------------------------------------------------
    # Remove images that are incorrectly classified
    predictions = nnutil.classify(nn, batch, batchsize, apply_softmax=False, progressbar=True)
    correct_preds = nnutil.prediction_equals(predictions, targets)
    correct_indices = np.sum(correct_preds, axis=0) == 2

    batch = batch[correct_indices]
    targets = targets[correct_indices]

    print('Remaining samples after filter: %d' % batch.shape[0])

    # ------------------------------------------------------------------------------------------------------------------
    # Classify once and save predictions to file
    predictions = nnutil.classify(nn, batch, batchsize, apply_softmax=True, progressbar=True)
    np.savez(os.path.join(DATA_OUT, 'data-predictions.npz'),
             **{args.categories[0]: predictions[0], args.categories[1]: predictions[1]})
    np.savez(os.path.join(DATA_OUT, 'data-targets.npz'),
             **{args.categories[0]: targets[:, 0], args.categories[1]: targets[:, 1]})

    if args.method.lower() == 'marginalize':
        # --------------------------------------------------------------------------------------------------------------
        # Estimate what the predictions of our model would look like if a feature would not exist/is unknown,
        # by marginalizing out this feature.
        # Refer to the paper for more info.
        print('Marginalizing out features...')
        for layer in args.layers:
            # calculate priors for each unit in layer
            prior_pdfs = surgery.parameter_priors(nn, layer)
            conditionals = surgery.marginalize_out_feature(nn,
                                                           batch,
                                                           batchsize,
                                                           layer,
                                                           prior_pdfs,
                                                           args.samples)

            np.savez_compressed(os.path.join(DATA_OUT, 'data-conditionals_layer=%s.npz' % layer),
                                **{args.categories[0]: conditionals[0],
                                   args.categories[1]: conditionals[1]})
    else:
        # --------------------------------------------------------------------------------------------------------------
        # Lesioning by setting parameters to 0, to determine each features impact to target output
        print('Lesioning...')
        for layer in args.layers:
            activations = surgery.volatile_lesioning(nn,
                                                     batch,
                                                     batchsize,
                                                     layer,
                                                     args.softmax)
            np.savez_compressed(os.path.join(DATA_OUT, 'data-lesionedpredictions_layer=%s.npz' % layer),
                                **{args.categories[0]: activations[0],
                                   args.categories[1]: activations[1]})

    print('Done')
