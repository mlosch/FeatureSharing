import numpy as np


def correction(p, n=1e6):
    """LaPlace correction
    """
    return (p*n+1) / (n*p.shape[1])


def odds(p):
    p = correction(p)
    return p / (1-p)


def WE(preds, conds):
    """Weighted evidence
    measured in bits
    """
    return np.log2(odds(preds)) - np.log2(odds(conds))


if __name__ == '__main__':
    from argparse import ArgumentParser
    from featuresharing import dataio
    from tqdm import tqdm
    import os

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path to data. Automatically looks for files:\n'
                                                                   'data-predictions.npz, data-targets.npz')
    parser.add_argument('--layers', type=str, required=True, nargs='+',
                        help='Layers to process. Automatically looks for files that match:\n'
                             'data-conditionals_layer=[layer].npz')
    parser.add_argument('--method', type=str, required=True,
                        help='Legal values: we, probdiff, infdiff')

    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate arguments

    methods = {
        'we': WE,
        'probdiff': lambda px, py: px-py,
        'infdiff': lambda px, py: np.log2(px) - np.log2(py),
    }

    if args.method.lower() not in methods:
        parser.error('Illegal method %s' % args.method)

    method = methods[args.method.lower()]

    # ------------------------------------------------------------------------------------------------------------------
    # Load data

    predictions, _ = dataio.loadnpz(os.path.join(args.datadir, 'data-predictions.npz'))

    # ------------------------------------------------------------------------------------------------------------------
    # Read and evaluate conditionals

    for layer in tqdm(args.layers):
        conditionals, keys = dataio.loadnpz(os.path.join(args.datadir, 'data-conditionals_layer=%s.npz' % layer))

        evidence = {}
        for i, key in enumerate(keys):
            evidence[key] = np.zeros(conditionals[i].shape)

            for unitidx in range(conditionals[i].shape[0]):
                evidence[key][unitidx, :] = method(predictions[i], conditionals[i][unitidx])

        np.savez(os.path.join(args.datadir, 'data-featurecontribution_layer=%s.npz' % layer), **evidence)

    print('Done')
