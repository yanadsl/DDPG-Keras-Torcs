from gym_torcs import TorcsEnv
import numpy as np
from qLearning import QL
import os

# OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=1):  # 1 means Train, 0 means simply Run
    actions = ['left', 'go', 'right']
    learning_rate = 0.2
    greedy = 0.1
    decay = 0.9

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
        # s_t = int(dis[0]+dis[10]+dis[18])

        s_t = normalize(ob.track)
        total_reward = 0.
        for j in range(max_steps):
            print(ob.track)
            action = Qlearning.action_choose(s_t)
            if action == 'left':
                actual_action = [-0.3, 1, 0]
                # actual_action = {'steer':'-0.3', 'acc':'1', 'brake':'0'}
            elif action == 'go':
                actual_action = [0, 1, 0]
                # actual_action = {'steer': '0', 'acc': '1', 'brake': '0'}
            else:
                actual_action = [0.3, 1, 0]
                # actual_action = {'steer': '0.3', 'acc': '1', 'brake': '0'}

            ob, r_t, done, info = env.step(actual_action)


            # s_t1 = int(dis[0] + dis[10] + dis[18])
            s_t1 = normalize(ob.track)
            if train_indicator:
                Qlearning.learn(s_t, action, r_t, s_t1, done)

            total_reward += r_t


            print("Episode", i, "State", s_t, "Action", action, "Reward", r_t)
            s_t = s_t1
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

def normalize(track):
    dis = ''
    for distance in track:
        if distance >= 36:
            dis += '8'
        elif distance >= 24:
            dis += '7'
        elif distance >= 20:
            dis += '6'
        elif distance >= 16:
            dis += '5'
        elif distance >= 12:
            dis += '4'
        elif distance >= 10:
            dis += '3'
        elif distance >= 8:
            dis += '2'
        elif distance >= 6:
            dis += '1'
        else:
            dis += '0'
    return dis[3]+dis[9]+dis[16]

if __name__ == "__main__":
    playGame()
