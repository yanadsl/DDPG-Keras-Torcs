from gym_torcs import TorcsEnv
import numpy as np
import random
import pandas
from qLearning import QL
import json

from ReplayBuffer import ReplayBuffer
from ActorNetwork import ActorNetwork
from CriticNetwork import CriticNetwork
from OU import OU
import timeit

OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    actions = ['left', 'go', 'right']
    learning_rate = 0.1
    greedy = 0.1
    decay = 0.2

    np.random.seed(1337)

    episode_count = 2000
    max_steps = 100000
    reward = 0
    done = False
    step = 0
    epsilon = 1

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    Qlearning = QL(actions, decay, greedy, learning_rate)

    # Now load the weight
    try:
        Qlearning.load("Qtable.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("TORCS Experiment Start.")
    for i in range(episode_count):

        if np.mod(i, 30) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()

        s_t = ob.trackPos

        total_reward = 0.
        for j in range(max_steps):

            action = Qlearning.action_choose(s_t)
            if action == 'left':
                actual_action = {'steer':'-0.3', 'acc':'1', 'brake':'0'}
            elif action == 'go':
                actual_action = {'steer': '0', 'acc': '1', 'brake': '0'}
            elif action == 'right':
                actual_action = {'steer': '0.3', 'acc': '1', 'brake': '0'}
            ob, r_t, done, info = env.step(actual_action)

            s_t1 = ob.trackPos

            if train_indicator:
                Qlearning.learn(s_t, action, r_t, s_t1, done)

            total_reward += r_t
            s_t = s_t1

            print("Episode", i, "Step", step, "Action", action, "Reward", r_t)

            step += 1
            if done:
                break

        if np.mod(i, 3) == 0:
            if train_indicator:
                print("Now we save model")
                Qlearning.save("Qtable.h5")

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # This is for shutting down TORCS
    print("Finish.")


if __name__ == "__main__":
    playGame()
