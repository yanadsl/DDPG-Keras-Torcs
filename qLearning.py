import numpy as np
import pandas as pd
import colorama
from colorama import Fore, Back, Style


class QL:
    def __init__(self, actions, learning_rate, greedy, decay, Lambda):
        self.actions = actions
        self.lr = learning_rate
        self.learning_rate = learning_rate
        self.g = greedy
        self.greedy = greedy
        self.decay = decay
        self.table = pd.DataFrame(columns=self.actions)
        self.backtrace = []
        self.Lambda = Lambda

    def load(self, fname):
        try:
            self.table = pd.read_hdf(fname, 'table')
            print("load successfully")
        except:
            print("no file to load")

    def save(self, fname):
        self.table.to_hdf(fname, 'table')

    def action_choose(self, ob):
        zero = True
        self.ob_exist(ob)
        if np.random.uniform() > self.greedy:
            action = self.table.ix[ob, :]
            for num in action:
                if num == 0:
                    zero = True
                else:
                    zero = False
                    break
            if(zero): #everything zero
                print(Back.GREEN)
                print(action)
                print(Style.RESET_ALL)
            else:
                print("[Left: " + str(format(action[0], '.3f')) + "\tGo: " + str(format(action[1], '.3f')) + "\tRight: " + str(format(action[2], '.3f')) + "]")
                # print(action)
            action = action.reindex(np.random.permutation(action.index))
            action = action.argmax()
        else:
            action = np.random.choice(self.actions)
            print(Back.YELLOW + "random" + Style.RESET_ALL)
        return action

    def learn(self, state, action, reward, next_state, done=False):
        self.ob_exist(next_state)
        q_guess = self.table.ix[state, action]
        if done:
            q = reward
        else:
            # sarsa
            q = reward + self.decay * self.table.ix[next_state, action]

        self.table.ix[state, action] += self.learning_rate * (q - q_guess)


    def SARSA_learn(self, state, a, reward, next_state, next_action):
        q_guess = self.table.ix[state, a]
        q = reward + self.decay * self.table.ix[next_state, next_action]  # SARSA
        self.table.ix[state, a] += self.learning_rate * (q - q_guess)

    def ob_exist(self, state):
        if state not in self.table.index:
            self.table = self.table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.table.columns,
                    name=state,
                )
            )

    def parameter_change(self, step_sum_rate):
        if self.learning_rate > 0.3:
            self.learning_rate -= step_sum_rate
        else:
            self.learning_rate = 0.3
        print("learning_rate: " + str(self.learning_rate))


    def parameter_reset(self):
        self.learning_rate = 0.3
        self.greedy = 0.01
        # self.learning_rate = self.lr
        # self.greedy = self.g

    def parameter_set(self, lr, grdy, decay):
        self.learning_rate = lr
        self.greedy = grdy
        self.decay = decay
        # self.learning_rate = self.lr
        # self.greedy = self.g
