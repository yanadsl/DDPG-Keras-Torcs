from qLearning import QL
import pandas as pd

class qlearning_lambda(QL):

    def __init__(self, actions, learning_rate, greedy, decay, Lambda):
        super().__init__(actions, learning_rate, greedy, decay)

        self.backtrace = self.table.copy()
        self.Lambda = Lambda

    def learn(self, state, action, reward, next_state, done=False):
        self.ob_exist(next_state)
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
            new = self.table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.table.columns,
                    name=state,
                )
            )
            self.table = self.table.append(new)
            self.backtrace = self.backtrace.append(new)

