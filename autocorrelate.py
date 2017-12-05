import sys
import pickle
import numpy as np
import pandas as pd

#import warningsfix


def autocorr1(mat):
    cor = np.corrcoef(mat)
    (rows, cols) = cor.shape
    gap = 1
    num_vals = rows - gap
    vals = np.zeros(num_vals)
    for i in range(num_vals):
        j = i + gap
        val = cor[i, j]
        #print(val)
        vals[i] = val
    return np.nanmean(vals)


def main(fn):
    with open(fn, 'rb') as f:
        runs_data = pickle.load(f)

    SAMPLE_LENGTHS = [1, 5, 20, 100, 500, 1000]
    SAMPLE_INTERVALS = [50, 100, 250, 500, 1000, 1500, 2000, 2500]
    SAMPLES = 10
    WARM_UP_TIME = 1000

    autocor_mat = np.full((len(SAMPLE_INTERVALS), len(SAMPLE_LENGTHS)), -1, dtype=float)
    (num_samples, num_simulations) = runs_data.shape

    for slength_i, sample_length in enumerate(SAMPLE_LENGTHS):
        for sinterval_i, sample_interval in enumerate(SAMPLE_INTERVALS):
            #print(sample_interval, sample_length)
            if sample_interval < sample_length:
                #print('break')
                continue
            # Sample 10 times
            variant_data = []
            for simulation_i in range(num_simulations):
                #print('simulation', simulation_i)
                samples = []
                for sample_i in range(SAMPLES):
                    sample_start = WARM_UP_TIME + sample_interval * sample_i
                    samples.append(np.mean(runs_data[
                        sample_start:sample_start + sample_length, simulation_i]))
                variant_data.append(samples)
            variant_data = np.matrix(variant_data)
            cor = autocorr1(variant_data)
            print(sinterval_i, slength_i, cor)
            print(cor)
            autocor_mat[sinterval_i, slength_i] = cor

    print(autocor_mat)
    df = pd.DataFrame(autocor_mat, index=SAMPLE_INTERVALS, columns=SAMPLE_LENGTHS)
    html = df.to_html()
    with open("autocor1.html", "w") as f:
        f.write(html)


if __name__ == '__main__':
    main(sys.argv[1])
