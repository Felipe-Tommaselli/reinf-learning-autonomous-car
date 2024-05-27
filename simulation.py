import sys
import numpy as np
import math
import random
import csv

import gymnasium as gym
import gym_env


def simulate(env, csv_file_path, MAX_EPISODES=9999, MAX_TRY=1000, learning_rate=0.3, gamma=0.6, epsilon=0.6, epsilon_decay=0.999):

    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table = {'v': np.zeros(num_box + (3,)), 'w': np.zeros(num_box + (3,))}
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(seed=episode)[0]
        total_reward = 0
        action = {'v': 0, 'w': 0}

        # count random actions
        crandom  = 0
        clearned = 0

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # AI tries up to MAX_TRY times
            for t in range(MAX_TRY):
                # In the beginning, do random action to learn
                if random.uniform(0, 1) < epsilon:
                    crandom += 1
                    action['v'] = env.action_space.sample()[0] 
                    action['w'] = env.action_space.sample()[1] 
                    action['v'] = np.select([action['v'] < 3, action['v'] >= 6], [0, 2], default=1)
                    action['w'] = np.select([action['w'] < -15, action['w'] > 15], [0, 2], default=1)
                else:
                    clearned += 1
                    action['v'] = np.argmax(q_table['v'][state])
                    action['w'] = np.argmax(q_table['w'][state])

                # Do action and get result
                next_state, reward, done, truncated, _ = env.step([action['v'], action['w']])
                total_reward += reward

                # Update Q values for both Q tables (2)
                for qt, a in [(q_table['v'], action['v']), (q_table['w'], action['w'])]:
                    q_value = qt[state][a]
                    best_q = np.max(qt[next_state])
                    qt[state][a] = (1 - learning_rate) * q_value + learning_rate * (reward + gamma * best_q)

                # Set up for the next iteration
                state = next_state

                # Draw games
                env.render()

                # When episode is done, print reward
                if done or t >= MAX_TRY - 1:
                    writer.writerow([episode, t, reward, total_reward])
                    print(f"Ep: {episode} ... Time Steps: {t} ... Total reward = {total_reward:.2f} ... randomness: {crandom/(crandom + clearned):.2f}")
                    break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay