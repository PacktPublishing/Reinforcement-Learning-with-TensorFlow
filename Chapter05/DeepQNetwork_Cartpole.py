
# coding: utf-8
import numpy as np
import gym
from Deep_Q_Network_Mountain_Car import DQN

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)


dqn = DQN(learning_rate=0.01,gamma=0.9,n_features=env.observation_space.shape[0],n_actions=env.action_space.n,epsilon=0.0,parameter_changing_pointer=100,memory_size=2000)

episodes = 150
total_steps = 0
rew_ep = []
for episode in range(episodes):
    steps = 0
    obs = env.reset()
    episode_reward = 0
    while True:
        env.render()
        action = dqn.epsilon_greedy(obs)
        obs_,reward,terminate,_ = env.step(action)

        #smaller the theta angle and closer to center then better should be the reward
        x, vel, angle, ang_vel = obs_
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(angle))/env.theta_threshold_radians - 0.5
        reward_ = r1 + r2

        dqn.store_experience(obs,action,reward_,obs_)
        if total_steps > 1000:
            dqn.fit()
        episode_reward+=reward
        if terminate:
            break
        obs = obs_
        total_steps+=1
        steps+=1
    print("Episode {} with Reward : {} at epsilon {} in steps {}".format(episode+1,episode_reward,dqn.epsilon,steps))
    rew_ep.append(episode_reward)
print("Mean over last 100 episodes are: ",np.mean(rew_ep[50:]))

while True:  #to hold the render at the last step when Car passes the flag
    env.render()




