import numpy
import sys
import pickle
import matplotlib.pyplot as plt
from math import sqrt
from scipy import stats

BOOTSTRAP_SEED = 42
BOOTSTRAP_ITERS = 10000
#BOOTSTRAP_SAMPLE_SIZE = 1000


def normal_mean(data):
    numpy.random.seed(BOOTSTRAP_SEED)
    mu = data.mean()
    sigma = data.std()
    N_ = data.shape[0]
    std_err = sigma/sqrt(N_)
    interval = stats.norm.interval(0.95, loc=mu, scale=std_err)
    return (mu, sigma, std_err, interval)


def bootstrap_mean(data, confidence=0.95):
    means = numpy.zeros(BOOTSTRAP_ITERS)
    sample_size = len(data)
    for i in range(BOOTSTRAP_ITERS):
        samp = data.sample(sample_size, replace=True)
        means[i] = samp.mean()
    means.sort()
    mid = numpy.median(means)
    margin = (1 - confidence) / 2 * 100
    lower = numpy.percentile(means, margin)
    upper = numpy.percentile(means, 100 - margin)
    return (lower, mid, upper)


def main(fn, plot_f):
    with open(fn, 'rb') as f:
        N, experiments = pickle.load(f)
    plot_rows = len(experiments)
    plot_cols = len(experiments[0][1].columns)
    fig, axes = plt.subplots(
        nrows=plot_rows, ncols=plot_cols,
        sharex='col', sharey=True)
    for i, ((prep, recovery), data) in enumerate(experiments):
        label = 'p{}r{}'.format(prep, recovery)
        print('##', label)
        for j, col in enumerate(data.columns.values):
            print(col)
            col_data = data[col]
            bins = numpy.arange(0, col_data.max() + 2, 1)
            col_data.hist(ax=axes[i, j], bins=bins)
            bootstats = bootstrap_mean(col_data)
            print('{:.2f} {:.2f} {:.2f}'.format(*bootstats))
            #normstats = normal_mean(col_data)
            #print(normstats)
        axes[i, 0].set_ylabel(label, rotation=90, size='large')
    for j, col in enumerate(experiments[-1][1].columns.values):
        axes[-1, j].set_xlabel(col, rotation=0, size='large')
    plt.tight_layout()
    plt.savefig(plot_f)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
