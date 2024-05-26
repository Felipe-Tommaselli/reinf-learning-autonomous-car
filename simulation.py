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

        q_table0 = q_table[0]
        q_table1 = q_table[1]

        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            # AI tries up to MAX_TRY times
            for t in range(MAX_TRY):
                
                #TODO CHANGE Q LEARNING TO CONTINOUS
                # In the beginning, do random action to learn
                if random.uniform(0, 1) < epsilon:
                    crandom += 1
                    action0 = env.action_space.sample()[0] 
                    action1 = env.action_space.sample()[1] 
                else:
                    clearned += 1
                    action0 = np.argmax(q_table0[state])
                    action1 = np.argmax(q_table1[state])

                # Define the actions DISCRETE
                action0 = np.select([action0 < 3, action0 >= 6], [1, 9], default=5)
                action1 = np.select([action1 < -15, action1 > 15], [-30, 30], default=0)

                print(f'action0: {action0}, action1: {action1}')

                # Do action and get result
                next_state, reward, done, truncated, _ = env.step([action0, action1])
                total_reward += reward

                # Get correspond q value from state, action pair
                print(f'state: {state},\naction0: {action0},\naction1: {action1},\nnext_state: {next_state},\nqtable0: {q_table0[state]}')
                q_value0 = q_table0[state][action0]
                best_q0 = np.max(q_table0[next_state])

                q_value1 = q_table1[state][action1]
                best_q1 = np.max(q_table1[next_state])

                # Q(state, action) <- (1 - a)Q(state, action) + a(reward + rmaxQ(next state, all actions))
                q_table0[state][action0] = (1 - learning_rate) * q_value0 + learning_rate * (reward + gamma * best_q0)
                q_table1[state][action1] = (1 - learning_rate) * q_value1 + learning_rate * (reward + gamma * best_q1)

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