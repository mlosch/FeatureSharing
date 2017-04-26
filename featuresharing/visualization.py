import os
import numpy as np
from tqdm import tqdm
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
