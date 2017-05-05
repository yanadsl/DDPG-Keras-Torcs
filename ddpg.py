from gym_torcs import TorcsEnv
import numpy as np
from qLearning import QL
from colorama import Fore, Back, Style
import os

# OU = OU()  # Ornstein-Uhlenbeck Process


def playGame(train_indicator=0) : # 1
    if train_indicator == 0:
        print(Back.RED + "NO TRAINING" + Style.RESET_ALL)

    # means Train, 0 means simply Run
    actions = ['left', 'go', 'right']
    learning_rate = 0.3 # 0.3
    greedy = 0 # 0.1
    decay = 0.7 # 0.5
    np.random.seed(1337)

    episode_count = 2000
    max_steps = 30000
    reward = 0
    done = False
    best_step_change = False
    step_list = [0, 0, 0, 0, 0]
    step_count = 0
    step_sum = 0
    step_sum_rate = 0

    # Generate a Torcs environment
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    Qlearning = QL(actions, decay, greedy, learning_rate)

    # Now load the weight
    Qlearning.load("Qtable.h5")

    try:
        file = open('episode.txt', 'r')
        episode_num = int(file.read())
        file.close()
        print("Episode number: " + str(episode_num))
    except:
        episode_num = 0
        print("Episode number: Error")

    try:
        file = open('best_step.txt', 'r')
        best_step = int(file.read())
        file.close()
        print("best_step: " + str(best_step))
    except:
        best_step = 0
        print("best_step: Error")

    print("TORCS Experiment Start.")
    for i in range(episode_num, episode_count):

        file = open('episode.txt', 'w')
        file.write(str(i))
        file.close()

        print("step_sum: "+str(step_sum))


        if step_sum < 200:
            Qlearning.parameter_set(0.5, 0.05, 0.7)
            print(Back.BLUE + "set1" + Style.RESET_ALL)
        elif step_sum < 400:
            Qlearning.parameter_set(0.5, 0.03, 0.7)
            print(Back.BLUE + "set2" + Style.RESET_ALL)
        elif step_sum < 600:
            Qlearning.parameter_set(0.5, 0.02, 0.7)
            print(Back.BLUE + "set3" + Style.RESET_ALL)
        elif step_sum < 800:
            Qlearning.parameter_set(0.5, 0.01, 0.7)
            print(Back.BLUE + "set4" + Style.RESET_ALL)
        elif step_sum < 5000:
            Qlearning.parameter_set(0.5, 0.005, 0.7)
        else:
            Qlearning.parameter_set(0.3, 0.001, 0.7)
            print(Back.BLUE + "set5" + Style.RESET_ALL)


        if np.mod(i, 30) == 0:
            ob = env.reset(relaunch=True)  # relaunch TORCS every 3 episode because of the memory leak error
        else:
            ob = env.reset()
        # s_t = int(dis[0]+dis[10]+dis[18])
        # os.system('sh speedup.sh')

        s_t = normalize(ob.track)
        total_reward = 0
        for j in range(max_steps):
            #if j > 0:
                #Qlearning.parameter_set(0.3, 0.01, 0.5)

            #Qlearning.parameter_change(step_sum_rate)

            print("[" + str(ob.track[3]) + " " + str(ob.track[9]) + " " + str(ob.track[16]) + "]")
            action = Qlearning.action_choose(s_t)
            if action == 'left':
                actual_action = [0.6, 0.1, 0]
                # actual_action = {'steer':'-0.3', 'acc':'1', 'brake':'0'}
            elif action == 'go':
                actual_action = [0, 1, 0]
                # actual_action = {'steer': '0', 'acc': '1', 'brake': '0'}
            else:
                actual_action = [-0.6, 0.1, 0]
                # actual_action = {'steer': '0.3', 'acc': '1', 'brake': '0'}

            ob, r_t, done, info = env.step(actual_action)

            ## if step <= 450:
                ## Qlearning.parameter_change(step)

            # s_t1 = int(dis[0] + dis[10] + dis[18])
            s_t1 = normalize(ob.track)
            if train_indicator:
                Qlearning.learn(s_t, action, r_t, s_t1, done)

            total_reward += r_t


            print("Episode", i,"step", j, "State", s_t, "Action", action, "Reward", r_t)
            s_t = s_t1
            if done:
                break

            if (j != 0) & (j % 384 == 0):
                print(Back.RED + "Qlearning table is saved" + Style.RESET_ALL)
                Qlearning.save("Qtable.h5")


            if j > best_step:
                best_step = j
                best_step_change = True
            else:
                best_step_change = False

        if best_step_change:
            #Qlearning.save("Qtable_"+str(best_step)+".h5")
            file = open('best_step.txt', 'w')
            file.write(str(best_step))
            file.close()
            print(Back.LIGHTGREEN_EX + "Best Step CHANGE!!!" + Style.RESET_ALL)

        if train_indicator:
            print(Back.RED + "Now we save model" + Style.RESET_ALL)
            Qlearning.save("Qtable.h5")

        print("TOTAL REWARD @ " + str(i) + "-th Episode  : Reward " + str(total_reward))
        print("Best Step: " + str(best_step))
        print("")

        step_list[step_count] = j
        if step_count == 4:
            step_count = 0
        else:
            step_count = step_count + 1

        step_sum = 0
        for number in step_list:
            step_sum += number
        step_sum = step_sum / 5
        step_sum_rate = 0.3 / step_sum

        score = open('score.txt', 'a')
        score.write("Episode: " + str(i) + "\tStep: " + str(j) + "\n")
        score.close()

    env.end()  # This is for shutting down TORCS
    print("Finish.")

def normalize(track):
    dis = []
    for distance in track:
        if distance >= 36:
            dis.append("16")
        elif distance >= 34:
            dis.append("15")
        elif distance >= 32:
            dis.append("14")
        elif distance >= 30:
            dis.append("13")
        elif distance >= 28:
            dis.append("12")
        elif distance >= 26:
            dis.append("11")
        elif distance >= 24:
            dis.append("10")
        elif distance >= 22:
            dis.append("09")
        elif distance >= 20:
            dis.append("08")
        elif distance >= 18:
            dis.append("07")
        elif distance >= 16:
            dis.append("06")
        elif distance >= 14:
            dis.append("05")
        elif distance >= 12:
            dis.append("04")
        elif distance >= 10:
            dis.append("03")
        elif distance >= 8:
            dis.append("02")
        elif distance >= 6:
            dis.append("01")
        elif distance >= 4:
            dis.append("01")
        else:
            dis.append("00")
        """
        if distance >= 36:
            dis.append("08")
        elif distance >= 24:
            dis.append("07")
        elif distance >= 20:
            dis.append("06")
        elif distance >= 16:
            dis.append("05")
        elif distance >= 12:
            dis.append("04")
        elif distance >= 10:
            dis.append("03")
        elif distance >= 8:
            dis.append("02")
        elif distance >= 6:
            dis.append("01")
        else:
            dis.append("00")
        """
    return dis[3]+dis[9]+dis[16]

if __name__ == "__main__":
    playGame()
