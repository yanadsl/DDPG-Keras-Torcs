from qLearning import QL
import pandas as pd
from colorama import Fore, Back, Style

class qlearning_lambda(QL):

    def __init__(self, actions, learning_rate, greedy, decay, Lambda):
        QL.__init__(self, actions, learning_rate, greedy, decay)

        self.backtrace = self.table.copy()
        self.Lambda = Lambda

    def load(self, fname):
        try:
            self.table = pd.read_hdf(fname, 'table')
            self.backtrace = self.table.copy()
            self.back_reset()
            print("load successfully")
        except:
            print("no file to load")

    def learn(self, state, action, reward, next_state, done=False):
        self.ob_exist(next_state)
        self.back_exist(next_state)
        q_guess = self.table.ix[state, action]
        if done:
            q = reward
        else:
            q = reward + self.decay * self.table.ix[next_state, action]

        # self.table.ix[state, action] += self.learning_rate * (q - q_guess)

        diff = q - q_guess

        self.backtrace.ix[state, :] *= 0
        self.backtrace.ix[state, action] = 1

        self.table += self.lr * diff * self.backtrace
        self.backtrace *= self.decay * self.Lambda

    def ob_exist(self, state):
        if state not in self.table.index:
            new = pd.Series(
                    [0] * len(self.actions),
                    index=self.table.columns,
                    name=state,
                )
            self.table = self.table.append(new)
            self.backtrace = self.backtrace.append(new)

    def back_exist(self, state):
        if state not in self.backtrace.index:
            new = pd.Series(
                    [0] * len(self.actions),
                    index=self.backtrace.columns,
                    name=state,
                )
            self.backtrace = self.backtrace.append(new)

    def back_reset(self):
        for act in self.actions:
            self.backtrace[act] = 0