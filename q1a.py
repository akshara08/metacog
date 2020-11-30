import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from Game.gamble import Gamble
from tqdm import tqdm


x = np.arange(199, 2202, 200)
num_samples = 100
freqs, perf = defaultdict(dict), defaultdict(dict)
colors = ['b', 'g', 'r', 'c', 'm']
markers = ["o", "^", "x", "+", "p" ]

for i in tqdm(range(num_samples)):
    g = Gamble()
    g.begin_trails()
    for ix, key in enumerate(sorted(g.freqs)):
        freqs[key][i] = g.freqs[key]
        perf[key][i] = g.performance[key]

means_freq, errors_freq = {}, {}
means_perf, errors_perf = {}, {}

for key in sorted(perf):
    dist_freq, dist_perf = [], []
    for ix in range(num_samples):
        dist_freq.append(freqs[key][ix])
        dist_perf.append(perf[key][ix])

    means_freq[key] = np.mean(dist_freq, axis=0)
    errors_freq[key] = np.std(dist_freq, axis=0, ddof=1)

    means_perf[key] = np.mean(dist_perf, axis=0)
    errors_perf[key] = np.std(dist_perf, axis=0, ddof=1)

#1a - During Training Learning
labels = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1']
for ix, key in enumerate(sorted(means_freq)):
    plt.plot(x, means_freq[key], marker=markers[ix], label=labels[ix], color=colors[ix])
    plt.errorbar(x, means_freq[key], yerr=errors_freq[key], fmt='o', capsize=5, color=colors[ix])

plt.legend(loc=2)
plt.xlabel("Number of Trials")
plt.ylabel("Cummulative Ratio of choosing LEX over EQW")
plt.show()

for ix, key in enumerate(sorted(means_perf)):
    plt.plot(x, means_perf[key], marker=markers[ix], label=labels[ix], color=colors[ix])
    plt.errorbar(x, means_perf[key], yerr=errors_perf[key], fmt='o', capsize=5, color=colors[ix])

plt.legend(loc=2)
plt.xlabel("Number of Trials")
plt.ylabel("Absolute Difference between the q-values of LEX and EQW")
plt.show()
print()



