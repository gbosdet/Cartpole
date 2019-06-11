import gym
import numpy as np

env = gym.make("CartPole-v0")
observation = env.reset()

box = env.observation_space

done = False

action = 0
weights = np.zeros(4)
weight_sum = 0
print(weights)
for trial in range(1000):
    sum = 0
    current_weights = np.random.randn(4)
    for episode in range(40):
        count = 0
        while not done:
            if observation.dot(weights) >= 0:
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
        print("Weights: " + str(weights))
        print("Sum: " + str(weight_sum / 40))
print("Weights: " + str(weights))
print("Sum: " + str(weight_sum / 40))
