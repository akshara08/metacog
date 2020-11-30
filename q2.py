import numpy as np
from Game.gamble import Gamble
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_1samp

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

num_samples = 200
proportion = []
labels = ["1", "2", "3", "4", "5", "6"]
x = np.arange(len(labels))
width = 0.35

#run experiment 200 times for p=0.6
for i in tqdm(range(num_samples)):
    g = Gamble()
    g.begin_trails(n=140, p=0.6, checkpoint=20)
    assert len(list(g.prop))==1
    for ix, key in enumerate(sorted(g.prop)):
        proportion.append(g.prop[key])

means6 = np.round(np.mean(proportion, axis=0), 2)
errs6 = np.round(np.std(proportion, axis=0, ddof=1), 2)
hum6 = [0.4, 0.55, 0.65, 0.70, 0.75, 0.80]

#plot bar
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means6, width, yerr=errs6, label='Agent')
rects2 = ax.bar(x + width/2, hum6, width, label='Human')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
ax.legend()
plt.xlabel("Blocks")
plt.ylabel("Percentage of EQW for p=0.6")
plt.ylim(0, 1)
plt.show()
pvals = []

#calculate p-values using one sample t-test
for ix in range(len(hum6)):
    dist = [samp[ix] for samp in proportion]
    tscore, pvalue = ttest_1samp(dist, popmean=hum6[ix])
    pvals.append(pvalue)

print("p=0.6", pvals)

#run experiment 200 times for p=0.9
proportion = []
for i in tqdm(range(num_samples)):
    g = Gamble()
    g.begin_trails(n=70, p=0.9, checkpoint=10)
    assert len(list(g.prop))==1
    for ix, key in enumerate(sorted(g.prop)):
        proportion.append(g.prop[key])

means9 = np.round(np.mean(proportion, axis=0), 2)
errs9 = np.round(np.std(proportion, axis=0, ddof=1), 2)
hum9 = [0.65, 0.5, 0.45, 0.40, 0.35, 0.30]

#plot bar
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, means9, width, yerr=errs9, label='Agent')
rects2 = ax.bar(x + width/2, hum9, width, label='Human')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
ax.legend()
plt.xlabel("Blocks")
plt.ylabel("Percentage of EQW for p=0.9")
plt.ylim(0, 1)
plt.show()
pvals = []

#calculate p-values using one sample t-test
for ix in range(len(hum9)):
    dist = [samp[ix] for samp in proportion]
    tscore, pvalue = ttest_1samp(dist, popmean=hum9[ix])
    pvals.append(pvalue)

print("p=0.9", pvals)

print()