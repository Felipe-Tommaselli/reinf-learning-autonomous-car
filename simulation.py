import sys
import numpy as np
import math
import random
import csv

import gymnasium as gym
import gym_env


def simulate(env, q_table, csv_file_path, MAX_EPISODES=9999, MAX_TRY=1000, learning_rate=0.3, gamma=0.6, epsilon=0.6, epsilon_decay=0.999):
    epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(seed=episode)[0]
        total_reward = 0

        # count random actions
        crandom  = 0
        clearned = 0 

        # save infos for analysis
        totalreward_list = list()
        reward_list = list()
        eps_list = list()
        actions_list = [0, 0, 0, 0]
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # AI tries up to MAX_TRY times
            for t in range(MAX_TRY):
                
                # In the beginning, do random action to learn
                if random.uniform(0, 1) < epsilon:
                    crandom += 1
                    action = env.action_space.sample()
                else:
                    clearned += 1
                    action = np.argmax(q_table[state])

                actions_list.pop(0)
                actions_list.append(action)

                # if actions list full of same actions, change last action randomly
                actions_list[-1] = random.randint(0, 2) if actions_list.count(actions_list[-1]) >= 3 else actions_list[-1]
                action = actions_list[-1]

                # Do action and get result
                next_state, reward, done, truncated, _ = env.step(action)
                total_reward += reward

                # Get correspond q value from state, action pair
                q_value = q_table[state][action]
                best_q = np.max(q_table[next_state])

                #TODO: Debug this
                # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
                q_table[state][action] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

                # Set up for the next iteration
                state = next_state

                # Draw games
                env.render()

                # save infos
                totalreward_list.append(total_reward)
                reward_list.append(reward)
                eps_list.append(t)

                # When episode is done, print reward
                if done or t >= MAX_TRY - 1:
                    writer.writerow([episode, t, reward, total_reward])
                    print(f"Ep: {episode} ... Time Steps: {t} ... Total reward = {total_reward:.2f} ... randomness: {crandom/(crandom + clearned):.2f}")
                    break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay