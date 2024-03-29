from collections import deque
import gym
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DDQN:
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes=[80, 80], activation_function="relu", optimizer="adam", loss_function="mean_squared_error", gamma=0.99, experience_buffer_size=20000, min_buffer_size=1000, learning_rate=0.001, training_batch_size=50, training_times_per_weight_copy=10):
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.gamma = gamma
        self.experience_buffer_size = experience_buffer_size
        self.min_buffer_size = min_buffer_size
        self.training_batch_size = training_batch_size
        self.training_times_per_weight_copy = training_times_per_weight_copy

        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_layer_sizes[0], input_shape=(num_inputs,), activation=activation_function))
        for i in range(1, len(hidden_layer_sizes)):
            self.model.add(tf.keras.layers.Dense(hidden_layer_sizes[0], activation=activation_function))
        self.model.add(tf.keras.layers.Dense(num_outputs, activation="linear"))
        self.model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss=loss_function)

        self.target_model = tf.keras.models.Sequential()
        self.target_model.add(tf.keras.layers.Dense(hidden_layer_sizes[0], input_shape=(num_inputs,), activation=activation_function))
        for i in range(1, len(hidden_layer_sizes)):
            self.target_model.add(tf.keras.layers.Dense(hidden_layer_sizes[0], activation=activation_function))
        self.target_model.add(tf.keras.layers.Dense(num_outputs, activation="linear"))
        self.target_model.compile(tf.keras.optimizers.Adam(lr=learning_rate), loss=loss_function)
        self.target_model.set_weights(self.model.get_weights())

        self.experiences = deque(maxlen=experience_buffer_size)
        self.training_iterations = 0

    def add_experience(self, step_data):
        if len(self.experiences) >= self.experience_buffer_size:
            self.experiences.popleft()
        self.experiences.append(step_data)
        print(len(self.experiences))


    def save_weights(self, filename):
        pass

    def get_action(self, state, random_action_rate):
        if np.random.random() < random_action_rate:
            return np.random.randint(0, self.num_outputs)
        else:
            return np.argmax(self.model.predict(state))

    def train(self):
        if len(self.experiences) >= self.min_buffer_size:
            indexes = np.random.choice(len(self.experiences), size=self.training_batch_size, replace=False)
            # state - 0, action - 1, reward - 2, next_state - 3, done - 4
            X = np.array([self.experiences[index][0] for index in indexes]).reshape(self.training_batch_size, self.num_inputs)
            #target is the reward if the episode just finished otherwise it is equal to the reward plus what the target network thinks you can get taking the best action in the next state
            y = [self.experiences[index][2] if self.experiences[index][4] else self.experiences[index][2] + self.gamma*np.max(self.target_model.predict(self.experiences[index][3])) for index in indexes]
            self.model.fit(X, y, batch_size=int(self.training_batch_size/2), verbose=0)
            self.training_iterations += 1
            if self.training_iterations % self.training_times_per_weight_copy == 0:
                self.target_model.set_weights(self.model.get_weights())

def run_episode(env, ddqn, random_action_rate):
    last_state = env.reset().reshape(1, 4)
    done = False
    reward_sum = 0
    iterations = 0
    while not done and iterations < 3000:
        action = ddqn.get_action(last_state, random_action_rate)
        next_state, reward, done, info = env.step(action)
        next_state = next_state.reshape(1, len(next_state))
        ddqn.add_experience((last_state, action, reward, next_state, done))
        last_state = next_state
        ddqn.train()
        reward_sum += reward
        iterations += 1
    return reward_sum, iterations

def learn_game():
    sns.set()
    random_action_rate = 0.4
    random_action_decay_rate = 0.002

    env = gym.make('CartPole-v0')
    ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
    rewards = []
    steps_to_finish = []
    for i in range(1000):
        random_action_rate = 1/np.sqrt(i+1)
        reward, steps = run_episode(env, ddqn, random_action_rate)
        rewards.append(reward)
        steps_to_finish.append(steps)
        #random_action_rate -= random_action_decay_rate
        if i % 20 == 0:
            print("Episode:", i, "\tReward:", reward, "\tSteps:", steps)


    plt.plot(rewards)
    plt.title("Rewards")
    plt.show()
    plt.plot(steps)
    plt.title("Steps")
    plt.show()

if __name__ == "__main__":
    learn_game()
