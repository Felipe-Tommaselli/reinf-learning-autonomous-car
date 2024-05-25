import sys
import numpy as np
import math
import random

import os
from datetime import datetime

import gymnasium as gym
import gym_env
import csv

from simulation import simulate

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
    simulate(env, q_table, csv_file_path, MAX_EPISODES, MAX_TRY, learning_rate, gamma, epsilon, epsilon_decay)
