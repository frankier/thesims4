import random
import numpy
from matplotlib import pyplot as plt
from scipy.stats import chisquare


SEED = 42


def mk_triangular(a, b, c):
    """
    Returns a random number according to a PDF with a triangular shape starting
    at a and ending at c, with a maximum at b.

    Since it's based on a generator from [0, 1), it generates [a, c).

    Uses 1 random number generation from underlying PRNG.
    """
    cutoff = (b - a) / (c - a)

    def inner():
        x = random.random()
        if x < cutoff:
            x2 = x / cutoff
            return a + (x2 ** 0.5) * (b - a)
        else:
            x2 = (x - cutoff) / (1 - cutoff)
            return b + (1 - (1 - x2) ** 0.5) * (c - b)
    return inner


def vis_triangular():
    t = mk_triangular(-1, 0, 2)
    plt.hist([t() for x in range(1000000)], bins=100)
    plt.savefig('triangular_m1_0_2.pdf')


class LehmerGen:
    def __init__(self, seed):
        self.X = seed

    def __next__(self):
        self.X = (self.A * self.X + self.C) % self.m
        return self.X / self.m


class EniacGen(LehmerGen):
    m = 10 ** 8 + 1
    A = 23
    C = 0


def vis_eniac():
    numbers = []
    r = EniacGen(SEED)
    for _ in range(10000):
        numbers.append(next(r))
    plt.figure()
    plt.plot(numbers, ',')
    plt.savefig('eniac1d.pdf')
    # Bad randomness 1: Can easily see pattern in this plot
    plt.figure()
    plt.plot(numbers[:-1:], numbers[1::], ',')
    plt.savefig('eniac2d.pdf')


def prev_cond_test(cond):
    numbers = []
    r = EniacGen(SEED)
    x_prev = None
    for _ in range(1000000):
        x = next(r)
        if x_prev is not None and cond(x_prev):
            numbers.append(x)
        x_prev = x
    return numbers


def cond_test():
    numbers = prev_cond_test(lambda x: x < 0.01)
    (hist, bin_edges) = numpy.histogram(numbers, range=(0, 1))
    print('Data binned into 10 bins.')
    print('Each bin should also be drawn from uniform distribution if whole '
          'distribution is random.')
    print(hist)
    print('Result from Chi-square test', chisquare(hist))
    print('If p value is small, we should reject the null hypothesis: and '
          'conclude bins not from same distribution => whole distribution is '
          'not uniform.')
    # Bad randomness 2: conditioned randomness not uniform
