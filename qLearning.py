import numpy as np
import pandas as pd


class QL:
    def __init__(self, actions, learning_rate, greedy, decay):
        self.actions = actions
        self.learning_rate = learning_rate
        self.greedy = greedy
        self.decay = decay
        self.table = pd.DataFrame(columns=self.actions)

    def load(self, fname):
        try:
            self.table = pd.read_hdf(fname)
            print("load successfully")
        except:
            print("no file to load")

    def save(self, fname):
        self.table.to_hdf(fname, 'table')

    def action_choose(self, ob):
        self.ob_exist(ob)
        if np.random.uniform() > self.greedy:
            action = self.table.ix[ob, :]
            print(action)
            action = action.reindex(np.random.permutation(action.index))
            action = action.argmax()
        else:
            action = np.random.choice(self.actions)
            print("random")
        return action

    def learn(self, state, action, reward, next_state, done = False):
        self.ob_exist(next_state)
        q_guess = self.table.ix[state, action]
        if done:
            q = reward
        else:
            q = reward + self.decay * self.table.ix[next_state, action]
        self.table.ix[state, action] += self.learning_rate * (q - q_guess)

    def ob_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.table.columns,
                    name=state,
                    )
            )
