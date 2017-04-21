import time

from RpiEnv import Env
import numpy as np
from qLearning import QL
import os

# OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    actions = ['left', 'go', 'right']
    learning_rate = 0.4
    greedy = 0.1
    decay = 0.9

    np.random.seed(1337)

    episode_count = 2000
    max_steps = 100000
    step = 0

    # Generate a Torcs environment
    env = Env()

    Qlearning = QL(actions, decay, greedy, learning_rate)

    # Now load the weight
    try:
        Qlearning.load("Qtable.h5")
        print("Weight load successfully")
    except:
        print("Cannot find the weight")

    print("Autocar Experiment Start.")
    for i in range(episode_count):

        state = env.get_respond()
        step = 0
        total_reward = 0
        for j in range(max_steps):
            action = Qlearning.action_choose(state)
            env.step(action)
            time.sleep(0.5)

            new_state = env.get_respond()
            reward, dead = env.get_reward(new_state)
            if train_indicator:
                Qlearning.learn(state, action, reward, new_state)

            total_reward += reward


            print("Episode", i, "State", state, "Action", action, "Reward", reward)
            if dead:
                env.end()

            state = new_state
            step += 1

        if np.mod(i, 3) == 0:
            if train_indicator:
                print("Now we save model")
                Qlearning.save("Qtable.h5")

        # print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Total Step: " + str(step))
        print("")

    env.end()  # Stop Servos
    print("Finish.")



if __name__ == "__main__":
    playGame()
