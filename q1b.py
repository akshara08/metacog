import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from Game.gamble import Gamble
from tqdm import tqdm

x = np.arange(199, 2202, 200)
num_samples = 100
colors = ['b', 'g', 'r', 'c', 'm']
markers = ["o", "^", "x", "+", "p" ]
freqs = defaultdict(dict)
means_freq, errors_freq = {}, {}

for i in tqdm(range(num_samples)):
    g = Gamble()
    g.begin_trails(a=0., b=1.)
    for ix, key in enumerate(sorted(g.freqs)):
        freqs[key][i] = g.freqs[key]



for key in sorted(freqs):
    dist_freq, dist_perf = [], []
    for ix in range(num_samples):
        dist_freq.append(freqs[key][ix])

    means_freq[key] = np.mean(dist_freq, axis=0)
    errors_freq[key] = np.std(dist_freq, axis=0, ddof=1)

clubbed_means, clubbed_stds = {}, {}
means, errs = [], []
for key in [0, 1, 2, 3, 4, 5]:
    if key <= 5:
        means.append(means_freq[key])
        errs.append(errors_freq[key])

clubbed_means[0] = np.mean(means, axis=0)
clubbed_means[1] = means_freq[9]

clubbed_stds[0] = np.mean(errs, axis=0)
clubbed_stds[1] = errors_freq[9]

labels = ['<=0.6', '0.9-1']
for ix in [0, 1]:
    plt.plot(x, clubbed_means[ix], marker=markers[ix], label=labels[ix], color=colors[ix])
    plt.errorbar(x, clubbed_means[ix], yerr=clubbed_stds[ix], fmt='o', capsize=5, color=colors[ix])

plt.legend(loc=2)
plt.xlabel("Number of Trials")
plt.ylabel("Ratio of choosing LEX over EQW")
plt.show()
print()

for i in tqdm(range(num_samples)):
    g = Gamble()
    g.begin_trails(a=0., b=1.)
    q_states = g.q_states
    g = Gamble(q_states)
    g.post_training_trials(a=0., b=1.)
    for ix, key in enumerate(sorted(g.freqs)):
        freqs[key][i] = g.freqs[key]

for key in sorted(freqs):
    dist_freq, dist_perf = [], []
    for ix in range(num_samples):
        dist_freq.extend(freqs[key][ix])

    means_freq[key] = np.mean(dist_freq)
    errors_freq[key] = np.std(dist_freq, ddof=1)

clubbed_means, clubbed_stds = [], []
means, errs = [], []
for key in [0, 1, 2, 3, 4, 5]:
    if key <= 5:
        means.append(means_freq[key])
        errs.append(errors_freq[key])

clubbed_means.append(np.mean(means))
clubbed_means.append(means_freq[9])

clubbed_stds.append(np.mean(errs, axis=0))
clubbed_stds.append(errors_freq[9])

labels = ['<=0.6', '0.9-1']
plt.bar(labels, clubbed_means, yerr=clubbed_stds)

plt.xlabel("Value of probability")
plt.ylabel("Percentage of choosing EQW")
plt.show()
print()