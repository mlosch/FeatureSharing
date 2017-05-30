import numpy as np
import matplotlib.pyplot as plt


def radius(k, n, b):
    if k > n - b:
        return 1
    return np.sqrt(k - 1 / 2.) / np.sqrt(n - (b + 1) / 2.)


def sunflower(n, alpha=0):
    coords = np.zeros((n, 2))
    b = np.floor(alpha * np.sqrt(n))

    phi = (np.sqrt(5) + 1) / 2
    for k in range(1, n + 1):
        r = radius(k, n, b)
        theta = 2. * np.pi * k / phi ** 2.
        coords[k - 1][0] = r * np.cos(theta)
        coords[k - 1][1] = r * np.sin(theta)

    return coords


def average_task_contribution(contributions, targets):
    N = targets[0].shape[0]

    # calculate expected contribution over distribution of inputs
    #  consider only the classes y that satisfy y == y_true
    avg_contribution = [e[:, np.arange(N), ts].mean(1) for e, ts in zip(contributions, targets)]
    # take the absolute
    avg_contribution = [np.abs(e) for e in avg_contribution]

    return avg_contribution


def plot_rectangle(filep, avg_contributions):
    output_shapes = {
        96: (12, 8),
        256: (32, 8),
        384: (32, 12),
        4096: (128, 32),
    }

    matshape = output_shapes[avg_contributions[0].shape[0]]

    # fig, axes = plt.subplots(1, len(mat_we), figsize=(10, 5))
    # for i, mat in enumerate(mat_we):
    #     ax = plt.subplot(1, len(mat_we), i + 1)
    #     im = ax.matshow(mat)
    #     ax.set_title(keys[i])
    #
    # cbar_ax = fig.add_axes([0.95, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    # # plt.savefig('plot_we_conv5.png', dpi=300)
    # plt.show()

    # interpret data as image
    im_we = [avg.reshape(*matshape)[..., np.newaxis] for avg in avg_contributions]
    im_we.reverse()
    im_we.insert(0, im_we[0])
    im_we = np.concatenate(im_we, axis=2)

    plt.figure(figsize=matshape)
    plt.imshow(im_we, interpolation='none', aspect='equal')
    ax = plt.gca()

    ax.set_yticks(np.linspace(-.5, matshape[0] - .5, matshape[0] + 1))
    ax.set_xticks(np.linspace(-.5, matshape[1] - .5, matshape[1] + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # get rid of little axes tick lines
    for tic in ax.xaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    for tic in ax.yaxis.get_major_ticks():
        tic.tick1On = tic.tick2On = False
    ax.grid(color='#999999', linestyle='-', linewidth=1)
    # set color of border
    plt.setp(ax.spines.values(), color='#999999')

    plt.savefig(filep, bbox_inches='tight', pad_inches=0)


def plot_sunflower(filep, avg_contributions):
    nunits = avg_contributions[0].shape[0]
    coords = sunflower(nunits)

    # interpret values as rgb colors
    im_we = [avg[..., np.newaxis] for avg in avg_contributions]
    im_we.reverse()
    im_we.insert(0, im_we[0])
    im_we = np.concatenate(im_we, axis=-1)

    plt.scatter(coords[:, 0], coords[:, 1], c=im_we, s=200.0)
    plt.axes().set_aspect('equal', 'datalim')

    plt.savefig(filep, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from featuresharing import dataio
    from tqdm import tqdm
    import os

    parser = ArgumentParser()
    parser.add_argument('--datadir', type=str, required=True, help='Path to data.')
    parser.add_argument('--layers', type=str, required=True, nargs='+',
                        help='Layers to process. Automatically looks for files that match:\n'
                             'data-featurecontribution_layer=[layer].npz')
    parser.add_argument('--method', type=str, required=True,
                        help='Legal values: rectangle, sunflower')
    parser.add_argument('--categories', type=str, nargs='+', required=True)
    parser.add_argument('--sortby', type=str, default='', help='Sort indices by category. By default does not sort.')
    parser.add_argument('--log', action='store_true', help='Use log transformation on color coding to brighten dark values.')

    args = parser.parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluate arguments

    methods = {
        'rectangle': plot_rectangle,
        'sunflower': plot_sunflower,
    }
    method = methods[args.method.lower()]

    # ------------------------------------------------------------------------------------------------------------------
    # Load data

    targets, keys = dataio.loadnpz(os.path.join(args.datadir, 'data-targets.npz'))

    for category in args.categories:
        if category not in keys:
            parser.error('Category %s does not exist in data-targets.npz')

    key_order = [keys.index(category) for category in args.categories]
    targets = [targets[i] for i in key_order]

    # ------------------------------------------------------------------------------------------------------------------
    # Read and visualize contributions

    for layer in tqdm(args.layers):
        contributions, keys = dataio.loadnpz(os.path.join(args.datadir, 'data-featurecontribution_layer=%s.npz' % layer))
        contributions = [contributions[i] for i in key_order]

        avg_contributions = average_task_contribution(contributions, targets)

        if args.log:
            for avg in avg_contributions:
                np.log(avg+1, avg)

        max_contribution = max([avg.max() for avg in avg_contributions])
        # cont = np.array(avg_contributions)
        # max_contribution = np.sqrt(np.sum(cont**2, 0))

        # Normalize to largest contribution
        avg_contributions = [avg/max_contribution for avg in avg_contributions]

        if args.sortby:
            inds = np.argsort(avg_contributions[args.categories.index(args.sortby)])
            avg_contributions = [avg[inds] for avg in avg_contributions]
            # np.save('data-ordered-indices_layer=%s.npy'%layer, inds)

        method(os.path.join(args.datadir, '{}_{}.png'.format('+'.join(args.categories), layer)), avg_contributions)


