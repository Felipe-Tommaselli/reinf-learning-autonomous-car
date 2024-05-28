import gymnasium as gym
import gym_env

from stable_baselines3 import SAC

def simulate(env, run_id, MAX_EPISODES=10000):
    # Instantiate the agent
    model = SAC(policy='MlpPolicy', env=env, verbose=1, 
                                            gamma=0.99, 
                                            learning_rate=0.0003, 
                                            buffer_size=50000, 
                                            learning_starts=100, 
                                            train_freq=1, 
                                            batch_size=64, 
                                            tau=0.005, 
                                            ent_coef='auto', 
                                            target_update_interval=1, 
                                            gradient_steps=1, 
                                            target_entropy='auto', 
                                            action_noise=None, 
                                            #random_exploration=0.0, 
                                            tensorboard_log=None, 
                                            _init_setup_model=True, 
                                            policy_kwargs=None, 
                                            #full_tensorboard_log=False, 
                                            seed=None, 
                                            #n_cpu_tf_sess=None
                                            )

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