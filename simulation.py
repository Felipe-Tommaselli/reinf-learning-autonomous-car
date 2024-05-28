import gymnasium as gym
import gym_env

def simulate(env_id, run_id, MAX_EPISODES=10000):
    # Instantiate the agent
    model = SAC('MlpPolicy', env, verbose=1)

    # Train the agent
    model.learn(total_timesteps=MAX_EPISODES * 1000)

    # Simulate and record results
    with open(f'runs/run_{run_id}/{run_id}.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        for episode in range(MAX_EPISODES):
            state = env.reset()
            total_reward = 0

            for t in range(1000):
                action, _states = model.predict(state, deterministic=True)
                state, reward, done, truncated, info = env.step(action)
                total_reward += reward

                env.render()

                if done:
                    writer.writerow([episode, t, reward, total_reward])
                    print(f"Ep: {episode} ... Time Steps: {t} ... Total reward = {total_reward:.2f}")
                    break