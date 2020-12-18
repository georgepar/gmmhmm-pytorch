from math import sqrt

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from gmm import DiagonalCovarianceGMM, FullCovarianceGMM

matplotlib.use('Agg')
sns.set(style="white", font="Arial")
colors = sns.color_palette("Paired", n_colors=12).as_hex()

DEVICE = "cuda"


def main():
    n, d = 300, 2

    # generate some data points ..
    data = torch.Tensor(n, d).normal_()
    # .. and shift them around to non-standard Gaussians
    data[:n//2] -= 1
    data[:n//2] *= sqrt(3)
    data[n//2:] += 1
    data[n//2:] *= sqrt(2)
    data = data.to(DEVICE)
    # Next, the Gaussian mixture is instantiated and ..
    n_components = 2
    model = FullCovarianceGMM(n_components, d, init="random", device=DEVICE).fit(data)
    # .. used to predict the data points as they where shifted
    logp = model(data)
    y = torch.max(logp, -1)[1].type(torch.LongTensor).cpu().numpy()
    data = data.cpu().numpy()
    plot(data, y)


def plot(data, y):
    n = y.shape[0]

    fig, ax = plt.subplots(1, 1, figsize=(1.61803398875*4, 4))
    ax.set_facecolor('#bbbbbb')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    # plot the locations of all data points ..

    for i, point in enumerate(data):
        if i <= n//2:
            # .. separating them by ground truth ..
            ax.scatter(*point, color="#000000", s=3, alpha=.75, zorder=n+i)
        else:
            ax.scatter(*point, color="#ffffff", s=3, alpha=.75, zorder=n+i)

        if y[i] == 0:
            # .. as well as their predicted class
            ax.scatter(*point, zorder=i, color="#dbe9ff", alpha=.6, edgecolors=colors[1])
        else:
            ax.scatter(*point, zorder=i, color="#ffdbdb", alpha=.6, edgecolors=colors[5])

    handles = [plt.Line2D([0], [0], color='w', lw=4, label='Ground Truth 1'),
        plt.Line2D([0], [0], color='black', lw=4, label='Ground Truth 2'),
        plt.Line2D([0], [0], color=colors[1], lw=4, label='Predicted 1'),
        plt.Line2D([0], [0], color=colors[5], lw=4, label='Predicted 2')]

    legend = ax.legend(loc="best", handles=handles)

    plt.tight_layout()
    plt.savefig("example.pdf")


if __name__ == "__main__":
    main()
