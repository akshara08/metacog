import numpy as np
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from sklearn.preprocessing import MaxAbsScaler


class Gamble:
    def __init__(self, q_states=None):
        self.actions = {0: self.LEX,
                        1: self.EQW}
        if q_states is None:
            self.q_states = np.zeros((10, 2))
        else:
            self.q_states = q_states
        self.action_times = defaultdict(lambda : defaultdict(float))
        self.proportions = defaultdict(lambda : defaultdict(float))
        self.times = Counter()
        self.exp_rate = 0.99
        self.lr = 0.99
        self.total_reward = 0


    def get_prob(self, a=0.5, b=1):
        return np.random.uniform(a, b)

    def payoffs(self):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        return x, y

    def LEX(self):
        if self.x1 > self.x2:
            return [self.x1, self.y1]
        else:
            return [self.x2, self.y2]

    def EQW(self):
        if (self.x1 + self.y1) > (self.x2 + self.y2):
            return [self.x1, self.y1]
        else:
            return [self.x2, self.y2]

    def state_idx(self, p):
        return math.floor(p*10)

    def random_action(self):
        return np.random.choice([0, 1])

    def result(self, x, y):
        return np.random.choice([x, y], p=[self.p, 1 - self.p])


    def choose_action(self, state_idx):
        # explore
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = self.random_action()
        else:
            # exploit
            action = np.argmax(self.q_states[state_idx])
        return action

    def take_action(self, action, state_idx, update=True):
        self.times[state_idx] += 1
        self.action_times[state_idx][action] += 1
        self.proportions[state_idx][action] += 1
        vars = self.actions[action]()
        feedback = self.result(vars[0], vars[1])
        self.total_reward += feedback
        if update == True:
            self.q_states[state_idx][action] += (self.lr*feedback)

    def begin_trails(self, n=2202, a=0.5, b=1, p=None, checkpoint=200):
        self.freqs = defaultdict(list)
        self.performance = defaultdict(list)
        self.scores = defaultdict(list)
        self.prop = defaultdict(list)
        self.leq_freq, self.eqw_freq = 0, 0
        for ix in range(1, n):
            if p == None:
                self.p = self.get_prob(a, b)
            else:
                self.p = p
            self.x1, self.y1 = self.payoffs()
            self.x2, self.y2 = self.payoffs()
            action = self.choose_action(self.state_idx(self.p))
            self.take_action(action, self.state_idx(self.p))
            if ix%checkpoint == 0:
                for state_ix in sorted(self.action_times):
                    self.freqs[state_ix].append((self.action_times[state_ix][0]+1)/(self.action_times[state_ix][1]+1))
                    self.performance[state_ix].append(abs(self.q_states[state_ix][0]-self.q_states[state_ix][1]))
                    self.prop[state_ix].append((self.proportions[state_ix][1]) / (self.proportions[state_ix][1] + self.proportions[state_ix][0]))
                    self.proportions[state_ix][0] = 0.
                    self.proportions[state_ix][1] = 0.
            if ix%(checkpoint//2) == 0:
                self.exp_rate = 1 - (ix/n)
                self.lr = 1 - (ix/n)

    def post_training_trials(self, n=100, a=0.5, b=1):
        self.exp_rate = 0.2
        self.freqs = defaultdict(list)
        for ix in range(1, n):
            self.p = self.get_prob(a, b)
            self.x1, self.y1 = self.payoffs()
            self.x2, self.y2 = self.payoffs()
            action = self.choose_action(self.state_idx(self.p))
            self.take_action(action, self.state_idx(self.p), update=False)
        for state_ix in sorted(self.action_times):
            self.freqs[state_ix].append((self.action_times[state_ix][1]) / (self.action_times[state_ix][1] + self.action_times[state_ix][0]))





