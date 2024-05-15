import sys
import numpy as np
import math
import random

import gymnasium as gym
import gym_env

def simulate():
    global epsilon, epsilon_decay
    for episode in range(MAX_EPISODES):

        # Init environment
        state = env.reset(seed=episode)[0]
        total_reward = 0

        # AI tries up to MAX_TRY times
        for t in range(MAX_TRY):
            
            #TODO: Count random actions through time
            # In the beginning, do random action to learn
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

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

            # When episode is done, print reward
            if done or t >= MAX_TRY - 1:
                print("Episode %d finished after %i time steps with total reward = %f." % (episode, t, total_reward))
                break

        # exploring rate decay
        if epsilon >= 0.005:
            epsilon *= epsilon_decay


if __name__ == "__main__":
    env = gym.make("Pygame-v0")
    MAX_EPISODES = 9999
    MAX_TRY = 1000
    epsilon = 0.5
    epsilon_decay = 0.9 # 0.999
    learning_rate = 0.2
    gamma = 0.6
    num_box = tuple((env.observation_space.high + np.ones(env.observation_space.shape)).astype(int))
    #TODO: Debug this
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()
