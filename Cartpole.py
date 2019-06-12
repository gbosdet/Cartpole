import gym
import numpy as np

env = gym.make("CartPole-v0")


box = env.observation_space



action = 0
weights = np.zeros(4)
weight_sum = 0
print(weights)
for trial in range(1000):
    sum = 0
    current_weights = np.random.randn(4)
    for episode in range(40):
        count = 0
        done = False
        observation = env.reset()
        while not done:
            if observation.dot(current_weights) >= 0:
                action = 1
            else:
                action = 0
            observation, reward, done, info = env.step(action)
            count += 1
        sum += count

    if sum > weight_sum:
        weight_sum = sum
        weights = current_weights
    if trial % 50 == 0:
        print()
        print("Trial: " + str(trial))
        print("\tWeights: " + str(weights))
        print("\tSum: " + str(weight_sum))
print("Final Information")
print("\tWeights: " + str(weights))
print("\tSum: " + str(weight_sum / 40))
done = False
observation = env.reset()
while not done:
    if observation.dot(weights) >= 0:
        action = 1
    else:
        action = 0
    observation, reward, done, info = env.step(action)
    env.render()
