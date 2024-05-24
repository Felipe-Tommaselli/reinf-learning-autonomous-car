import sys
import numpy as np
import math
import random

import os
import csv
from datetime import datetime

import gymnasium as gym
import gym_env

def simulate(csv_file_path):
    global epsilon, epsilon_decay
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
                #TODO: Fix this (negative reward)
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


# Function to create a directory and CSV file for storing data
def create_run_directory_and_file(run_id):
    # Create the directory with the run_id
    directory = f"runs/run_{run_id}"
    os.makedirs(directory, exist_ok=True)

    # Create the CSV file inside the directory
    csv_file_path = os.path.join(directory, f"{run_id}.csv")
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header row to the CSV file (customize headers as needed)
        writer.writerow(["Episode", "Time Steps", "Reward", "Total Reward"])

    return csv_file_path

if __name__ == "__main__":
    csv_file_path = create_run_directory_and_file(datetime.now().strftime('%m%d_%H%M'))
    
    env = gym.make("Pygame-v0")
    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 0.6
    epsilon_decay = 0.999 # 0.999
    learning_rate = 0.3
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    #TODO: Debug this
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate(csv_file_path)
