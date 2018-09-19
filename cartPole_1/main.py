import numpy as np 
import gym

env = gym.make('CartPole-v0')

def run_episode(env, parameters):  
    observation = env.reset()
    totalreward = 0
    for _ in range(200):
        action = 0 if np.matmul(parameters,observation) < 0 else 1
        env.render()
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward

# Random search
# bestparams = None  
# bestreward = 0  
# for _ in range(10000):  
#     parameters = np.random.rand(4) * 2 - 1
#     reward = run_episode(env,parameters)
#     if reward > bestreward:
#         bestreward = reward
#         bestparams = parameters
#         # considered solved if the agent lasts 200 timesteps
#         if reward == 200:
#             break

# Hill-Climbing

noise_scaling = 0.1  
parameters = np.random.rand(4) * 2 - 1  
episodes_per_update = 2
bestreward = 0  
for _ in range(10000):  
    newparams = parameters + (np.random.rand(4) * 2 - 1)*noise_scaling

    reward = 0  
    for _ in range(episodes_per_update):  
        run = run_episode(env,newparams)
        reward += run

    if reward > bestreward:
        noise_scaling = 0.1
        bestreward = reward
        parameters = newparams
        if reward == 400:
            break
    else:
        noise_scaling += 0.05        

print(parameters)