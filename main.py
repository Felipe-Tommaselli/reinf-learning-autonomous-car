import sys
import numpy as np
import math
import random
import os
from datetime import datetime

import gymnasium as gym
import gym_env
import csv
import yaml

from simulation import simulate

def main():
    csv_file_path = create_run_directory_and_file(datetime.now().strftime('%m%d_%H%M'))
    config = get_configs('configs/config.yaml')

    env = gym.make("Pygame-v0")
    MAX_EPISODES  = config['MAX_EPISODES']
    MAX_TRY       = config['MAX_TRY']
    epsilon       = config['epsilon']
    epsilon_decay = config['epsilon_decay']
    learning_rate = config['learning_rate']
    gamma         = config['gamma']

    #TODO: Debug this
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    q_table0 = np.zeros(num_box + (3,)) #! change this
    q_table1 = np.zeros(num_box + (3,))
    simulate(env, [q_table0, q_table1], csv_file_path, MAX_EPISODES, MAX_TRY, learning_rate, gamma, epsilon, epsilon_decay)


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

def get_configs(path_config):
    with open(path_config, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    main()