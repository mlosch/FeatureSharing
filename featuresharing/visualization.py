import os
import numpy as np
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt


def visualize_features(args, nn, layer_names, output_folder):
    dpi = 80.0
    imwidth = 224.0

    for layer in args.layers:
        weights, bias = nn.get_layerparams(layer)

        inds = range(weights.shape[0])
        nrows = np.ceil(np.sqrt(len(inds)))
        ncols = np.ceil(len(inds) / nrows)
        w = np.ceil((imwidth / dpi) * max(nrows, ncols))

        plt.figure(figsize=(w, w))
        for ui in tqdm(range(0, len(inds), 128), desc='Layer ' + layer):
            unitinds = inds[ui:(ui + 128)]
            ims = nn.visualize(None, layer, unitinds)
            ims = ims.transpose(0, 2, 3, 1)

            for i in range(ims.shape[0]):
                ax = plt.subplot(nrows, ncols, ui + i + 1)
                im = ims[i]
                im -= im.min()
                im /= im.max()

                if not os.path.exists(os.path.join(output_folder, layer_names[layer])):
                    os.mkdir(os.path.join(output_folder, layer_names[layer]))
                plt.imsave(os.path.join(output_folder, layer_names[layer], '%04d.png' % (ui + i)), im)
                ax.imshow(im)
                ax.set_xticks([])
                ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, layer_names[layer] + '.png'))
        plt.close()


def plot_accuracies(filep, accuracies, labels, title, ylabel, xlabel):
    plt.figure()

    accuracies_mean = accuracies.mean(0)
    accuracies_std = accuracies.std(0)

    plt.plot(accuracies_mean[:, 0], label=labels[0])
    plt.plot(accuracies_mean[:, 1], label=labels[1])

    if accuracies.shape[0] > 1:
        plt.fill_between(range(accuracies_mean.shape[0]), accuracies_mean[:, 0] - accuracies_std[:, 0],
                         accuracies_mean[:, 0] + accuracies_std[:, 0])
        plt.fill_between(range(accuracies_mean.shape[0]), accuracies_mean[:, 1] - accuracies_std[:, 1],
                         accuracies_mean[:, 1] + accuracies_std[:, 1])

    # plt.plot((0, accuracies_mean.shape[0]), (accura
    # cies_mean[0, 0], accuracies_mean[0, 0]), 'b--', label='Unlesioned %s recognition'%args.categories[0])
    # plt.plot((0, accuracies_mean.shape[0]), (accuracies_mean[0, 1], accuracies_mean[0, 1]), 'g--', label='Unlesioned %s recognition'%args.categories[1])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.legend()
    plt.savefig(filep)
    plt.close()


def multiplot_accuracies(outputfilep, accuracies_by_experiment, layers, labels, p_threshold=1e-5):
    nexp = len(accuracies_by_experiment)
    nlayers = len(layers[0])
    plt.figure(figsize=(18, (nexp * 18.) / nlayers))

    null_hypotheses = {}

    for k, accuracies_by_layer in enumerate(accuracies_by_experiment):
        for i, y in enumerate(accuracies_by_layer):
            y = y.squeeze()
            is_nullH = False

            if y.ndim == 3:
                y_std = y.std(0)
                y_mean = y.mean(0)
                y = y_mean
                is_nullH = True

                null_hypotheses[i] = (y_mean, y_std)

            layer = layers[k][i]
            ax = plt.subplot(nexp, 7, k * nlayers + i + 1)

            x = np.linspace(0, 1, y.shape[0])

            if is_nullH:
                ax.fill_between(x, y[:, 0] - y_std[:, 0], y[:, 0] + y_std[:, 0], alpha=0.5)
                ax.fill_between(x, y[:, 1] - y_std[:, 1], y[:, 1] + y_std[:, 1], alpha=0.5)

            plotsA = ax.plot(x, y[:, 0], label=labels[0])
            plotsB = ax.plot(x, y[:, 1], label=labels[1])

            # if not is_nullH and null_hypotheses[i] is not None:
            #     y_mean, y_std = null_hypotheses[i]
            #     ax.plot(x, 1.0 + y[:, 0] - y_mean[:, 0], color=plotsA[0].get_color())
            #     ax.plot(x, 1.0 + y[:, 1] - y_mean[:, 1], color=plotsB[0].get_color())

            if not is_nullH and null_hypotheses[i] is not None:
                # mark x's that are significantly different from null hypothesis
                y_mean, y_std = null_hypotheses[i]
                pA = norm(y_mean[:, 0], y_std[:, 0]).pdf(y[:, 0])
                pB = norm(y_mean[:, 1], y_std[:, 1]).pdf(y[:, 1])

                sigA = pA < p_threshold
                sigB = pB < p_threshold

                pointsA = x[sigA]
                pointsB = x[sigB]

                ax.plot(pointsA, np.ones((pointsA.shape[0],)) + 0.06, '.', color=plotsA[0].get_color())
                ax.plot(pointsB, np.ones((pointsB.shape[0],)) + 0.03, '.', color=plotsB[0].get_color())

            if k == 0:
                plt.title(layer)
            if k < nexp - 1:
                ax.get_xaxis().set_ticks([])
            if i == 0 and k == nexp // 2:
                plt.ylabel('Top-1 Accuracy')
            elif k == nexp - 1 and i == nlayers // 2:
                plt.xlabel('Fraction of features removed')
            if i > 0:
                ax.get_yaxis().set_ticks([])
            if i == nlayers - 1 and k == nexp - 1:
                plt.legend(loc=3)
            plt.tight_layout()

    plt.savefig(outputfilep, dpi=300)
    # plt.show()
    plt.close()
