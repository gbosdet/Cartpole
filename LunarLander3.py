from collections import deque
import gym
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time

class DDQN:
    def __init__(self, num_inputs, num_outputs, hidden_layer_sizes=[80, 80], activation_function="relu", optimizer="adam", loss_function="mean_squared_error", gamma=0.99, experience_buffer_size=80000, min_buffer_size=1000, learning_rate=0.001, training_batch_size=64, training_times_per_weight_copy=50):
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


    def save_weights(self, filename):
        pass

    def get_action(self, state, random_action_rate):
        if np.random.random() < random_action_rate:
            return np.random.randint(0, self.num_outputs)
        else:
            return np.argmax(self.model.predict(state.reshape(1, len(state))))

    def train(self):
        if len(self.experiences) >= self.min_buffer_size:
            indexes = np.random.choice(len(self.experiences), size=self.training_batch_size, replace=False)
            # state - 0, action - 1, reward - 2, next_state - 3, done - 4
            X = np.array([self.experiences[index][0] for index in indexes]).reshape(self.training_batch_size, self.num_inputs)
            states = np.array([self.experiences[index][0] for index in indexes])
            next_states = np.array([np.zeros(self.num_inputs) if self.experiences[index][4] else self.experiences[index][3] for index in indexes])
            state_predicted_values = self.model.predict(states)
            next_state_predicted_values = self.target_model.predict(next_states)
            #target is the reward if the episode just finished otherwise it is equal to the reward plus what the target network thinks you can get taking the best action in the next state
            y = np.empty((self.training_batch_size, self.num_outputs))
            for i in range(self.training_batch_size):
                experience = self.experiences[indexes[i]]
                y[i] = state_predicted_values[i]
                y[i][experience[1]] = experience[2] if experience[4] else experience[2] + self.gamma*np.max(next_state_predicted_values[i])
            self.model.fit(X, y, batch_size=self.training_batch_size, verbose=0)
            self.training_iterations += 1
            if self.training_iterations % self.training_times_per_weight_copy == 0:
                self.target_model.set_weights(self.model.get_weights())

    def initial_train(self):

        indexes = np.array([i for i in range(len(self.experiences)-1, -1, -1)])
        # state - 0, action - 1, reward - 2, next_state - 3, done - 4
        X = np.array([self.experiences[index][0] for index in indexes]).reshape(len(self.experiences), self.num_inputs)
        states = np.array([self.experiences[index][0] for index in indexes])
#        next_states = np.array([np.zeros(self.num_inputs) if self.experiences[index][4] else self.experiences[index][3] for index in indexes])
        state_predicted_values = self.model.predict(states)
#        next_state_predicted_values = self.target_model.predict(next_states)
        #target is the reward if the episode just finished otherwise it is equal to the reward plus what the target network thinks you can get taking the best action in the next state
        y = np.empty((len(self.experiences), self.num_outputs))
        for i in range(len(self.experiences)):
            experience = self.experiences[indexes[i]]
            y[i] = state_predicted_values[i]
            y[i][experience[1]] = experience[2] if experience[4] else experience[2] + self.gamma*y[i-1][self.experiences[indexes[i-1]][1]]
        self.model.fit(X, y, batch_size=len(self.experiences), verbose=0)
        self.training_iterations += 1
        if self.training_iterations % 10 == 0:
            self.target_model.set_weights(self.model.get_weights())

def run_episode(env, ddqn, random_action_rate, show_video):
    last_state = env.reset()

    last_state = np.append(last_state, [0])
    done = False
    reward_sum = 0
    iterations = 0
    while not done and iterations < 1001:
        iterations += 1
        action = ddqn.get_action(last_state, random_action_rate)
        next_state, reward, done, info = env.step(action)
        reward2 = reward
        reward_sum += reward
        too_high = 1000 - iterations - next_state[1]*1000
        if iterations > 400 and too_high < 0:
            reward2 += too_high/25
        if np.abs(next_state[0]) >= 0.99:
            reward2 -= 150
        reward2 += 2*(last_state[1]-next_state[1]) + 2*(np.abs(last_state[0]) - np.abs(next_state[0]))
        if done:
            print("\t\tReward:", reward_sum, "\tIteration: ", iterations, "\tX:", next_state[0], "\tY:", next_state[1])
        next_state = np.append(next_state, [iterations/1000])
        if show_video:
            env.render()
       # next_state = next_state.reshape(1, len(next_state))
        ddqn.add_experience((last_state, action, reward2, next_state, done))
        last_state = next_state
        ddqn.train()


    return reward_sum, iterations

def reward_mod(reward, last_state, current_state):
    for i in range(4):
        reward += abs(last_state[i]) - abs(current_state[i])
    return reward

def learn_game():
    sns.set()
    random_action_rate = 0.6
    random_action_decay_rate = 0.99
    hidden_layer_df = pd.DataFrame()
    steps_hidden_layer_df = pd.DataFrame()

    for name, layers in zip(["one 64 layer", "one 128 layer", "two 64 layer"], [[64], [128], [64, 64]]):
        start = time.time()
        env = gym.make('LunarLander-v2')
        # ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
        ddqn = DDQN(9, env.action_space.n, hidden_layer_sizes=layers, gamma=0.99, training_times_per_weight_copy=50)
        #ddqn.add_experience(((0, 0, 0, 0, 0, 0, 1, 1), 0, 1000, (0, 0, 0, 0, 0, 0, 1, 1), True))
        # num_found = 0
        # while num_found < 200:
        #     current_run = []
        #     total_reward = 0
        #     last_step = env.reset()
        #     done = False
        #     while not done:
        #         action = np.random.randint(env.action_space.n)
        #         next_state, reward, done, info = env.step(action)
        #         total_reward += reward
        #         current_run.append((last_step, action, reward, next_state, done))
        #         last_step = next_state
        #     if total_reward >= 100:
        #         for experience in current_run:
        #             ddqn.add_experience(experience)
        #         num_found += 1
        #         print("Found", num_found, "\tReward:", total_reward)
        # with open("good_episodes.txt", "wb") as file:
        #     pickle.dump(ddqn.experiences, file)
     #    good_experiences = []
     #    with open("good_episodes.txt", "rb") as file:
     #
     #        temp = pickle.load(file)
     #        iteration = 0
     #        for experience in temp:
     #            experience = (np.append(experience[0], [iteration]), experience[1], experience[2], experience[3], experience[4])
     #            ddqn.add_experience(experience)
     #            iteration += 1
     #            if experience[4]:
     #                iteration = 0
     # #       good_experiences = deque([x for x in ddqn.experiences])
     #
     #    for i in range(200):
     #        ddqn.initial_train()
        rewards = []
        steps_to_finish = []
        for i in range(501):
            #random_action_rate = 1/np.sqrt(i+1)
            reward, steps = run_episode(env, ddqn, random_action_rate, (i%20==0))
            rewards.append(reward)
            steps_to_finish.append(steps)
            hidden_layer_df.loc[i, name] = reward
            steps_hidden_layer_df.loc[i, name] = steps
            random_action_rate *= random_action_decay_rate
            if i % 20 == 0:
                print("\tEpisode:", i, "\tReward:", reward, "\tSteps:", steps)
            # if i % 500 == 0:
            #     # temp = ddqn.experiences
            #     # ddqn.experiences = good_experiences
            #     # for j in range(200):
            #     #     ddqn.initial_train()
            #     # ddqn.experiences = temp
            #     # for experience in good_experiences:
            #     #     ddqn.add_experience(experience)
            #     plt.plot(rewards)
            #     plt.title("Reward per Episode")
            #     plt.ylabel("Episode Reward")
            #     plt.xlabel("Episode")
            #     plt.savefig(("Reward vs Episode " + str(i)))
            #     plt.tight_layout()
            #     plt.close()
            #     plt.plot(steps_to_finish)
            #     plt.title("Steps per Episode")
            #     plt.ylabel("Episode Reward")
            #     plt.xlabel("Episode")
            #     plt.tight_layout()
            #     plt.savefig(("Steps vs Episode " + str(i)))
            #     plt.close()
        print(name, ":", (time.time()-start))



    hidden_layer_df.plot()
    plt.title("Reward per Episode")
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.tight_layout()
    plt.savefig(("./Figures/Hidden Layers Reward vs Episode 2 3"))
    plt.close()

    steps_hidden_layer_df.plot()
    plt.title("Hidden Layers Steps per Episode")
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.tight_layout()
    plt.savefig("./Figures/Steps vs Episode 2 3")
    plt.close()

    hidden_layer_df.to_csv("./Data/Hidden Layer Experiment 2 3.csv")
    steps_hidden_layer_df.to_csv("./Data/Hidden Layer Steps 2 3.csv")

def random_action_experiement():
    sns.set()
    #random_action_rate = 0.6
    random_action_decay_rate = 0.99
    hidden_layer_df = pd.DataFrame()
    steps_hidden_layer_df = pd.DataFrame()

    for name, random_action_rate in zip(["Initial RAR 0.3", "Initial RAR 0.4", "Initial RAR 0.5", "Initial RAR 0.6", "Initial RAR 0.7"], [0.3, 0.4, 0.5, 0.6, 0.7]):
        start = time.time()
        env = gym.make('LunarLander-v2')
        # ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
        ddqn = DDQN(9, env.action_space.n, hidden_layer_sizes=[64], gamma=0.99, training_times_per_weight_copy=50)
        #ddqn.add_experience(((0, 0, 0, 0, 0, 0, 1, 1), 0, 1000, (0, 0, 0, 0, 0, 0, 1, 1), True))
        # num_found = 0
        # while num_found < 200:
        #     current_run = []
        #     total_reward = 0
        #     last_step = env.reset()
        #     done = False
        #     while not done:
        #         action = np.random.randint(env.action_space.n)
        #         next_state, reward, done, info = env.step(action)
        #         total_reward += reward
        #         current_run.append((last_step, action, reward, next_state, done))
        #         last_step = next_state
        #     if total_reward >= 100:
        #         for experience in current_run:
        #             ddqn.add_experience(experience)
        #         num_found += 1
        #         print("Found", num_found, "\tReward:", total_reward)
        # with open("good_episodes.txt", "wb") as file:
        #     pickle.dump(ddqn.experiences, file)
     #    good_experiences = []
     #    with open("good_episodes.txt", "rb") as file:
     #
     #        temp = pickle.load(file)
     #        iteration = 0
     #        for experience in temp:
     #            experience = (np.append(experience[0], [iteration]), experience[1], experience[2], experience[3], experience[4])
     #            ddqn.add_experience(experience)
     #            iteration += 1
     #            if experience[4]:
     #                iteration = 0
     # #       good_experiences = deque([x for x in ddqn.experiences])
     #
     #    for i in range(200):
     #        ddqn.initial_train()
        rewards = []
        steps_to_finish = []
        for i in range(501):
            #random_action_rate = 1/np.sqrt(i+1)
            reward, steps = run_episode(env, ddqn, random_action_rate, (i%20==0))
            rewards.append(reward)
            steps_to_finish.append(steps)
            hidden_layer_df.loc[i, name] = reward
            steps_hidden_layer_df.loc[i, name] = steps
            random_action_rate *= random_action_decay_rate
            if i % 20 == 0:
                print("\tEpisode:", i, "\tReward:", reward, "\tSteps:", steps)
            # if i % 500 == 0:
            #     # temp = ddqn.experiences
            #     # ddqn.experiences = good_experiences
            #     # for j in range(200):
            #     #     ddqn.initial_train()
            #     # ddqn.experiences = temp
            #     # for experience in good_experiences:
            #     #     ddqn.add_experience(experience)
            #     plt.plot(rewards)
            #     plt.title("Reward per Episode")
            #     plt.ylabel("Episode Reward")
            #     plt.xlabel("Episode")
            #     plt.savefig(("Reward vs Episode " + str(i)))
            #     plt.tight_layout()
            #     plt.close()
            #     plt.plot(steps_to_finish)
            #     plt.title("Steps per Episode")
            #     plt.ylabel("Episode Reward")
            #     plt.xlabel("Episode")
            #     plt.tight_layout()
            #     plt.savefig(("Steps vs Episode " + str(i)))
            #     plt.close()
        print(name, ":", (time.time()-start))



    hidden_layer_df.plot()
    plt.title("Reward per Episode")
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.tight_layout()
    plt.savefig(("./Figures/RAR Reward vs Episode 2"))
    plt.close()

    steps_hidden_layer_df.plot()
    plt.title("RAR Steps per Episode")
    plt.ylabel("Episode Reward")
    plt.xlabel("Episode")
    plt.tight_layout()
    plt.savefig("./Figures/Steps vs Episode 2")
    plt.close()

    hidden_layer_df.to_csv("./Data/RAR Experiment 2.csv")
    steps_hidden_layer_df.to_csv("./Data/RAR Steps 2.csv")

def gamma_experiment():
    sns.set()
    random_action_rate = 0.6
    for trial in range(3, 5):
        random_action_decay_rate = 0.99
        hidden_layer_df = pd.DataFrame()
        steps_hidden_layer_df = pd.DataFrame()
        gammas = [0.95, 0.975, 0.99, 0.995]
        names = ["Gamma = 0.95", "Gamma = 0.975", "Gamma = 0.99", "Gamma = 0.995"]
        for name, gamma in zip(names, gammas):
            start = time.time()
            env = gym.make('LunarLander-v2')
            # ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
            ddqn = DDQN(9, env.action_space.n, hidden_layer_sizes=[64], gamma=gamma, training_times_per_weight_copy=50)

            rewards = []
            steps_to_finish = []
            for i in range(501):
                #random_action_rate = 1/np.sqrt(i+1)
                reward, steps = run_episode(env, ddqn, random_action_rate, (i%20==0))
                rewards.append(reward)
                steps_to_finish.append(steps)
                hidden_layer_df.loc[i, name] = reward
                steps_hidden_layer_df.loc[i, name] = steps
                random_action_rate *= random_action_decay_rate
                if i % 20 == 0:
                    print("\tEpisode:", i, "\tReward:", reward, "\tSteps:", steps)

            print(name, ":", (time.time()-start))



        hidden_layer_df.plot()
        plt.title("Gamma Reward per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig(("./Figures/Gamma Reward vs Episode " + str(trial)))
        plt.close()

        steps_hidden_layer_df.plot()
        plt.title("Gamma Steps per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig("./Figures/Gamma vs Episode " + str(trial))
        plt.close()

        hidden_layer_df.to_csv("./Data/Gamma Experiment " + str(trial) + ".csv")
        steps_hidden_layer_df.to_csv("./Data/Gamma Steps " + str(trial) + ".csv")


def learning_rate_experiment():
    sns.set()
    random_action_rate = 0.6
    for trial in range(3, 5):
        random_action_decay_rate = 0.99
        hidden_layer_df = pd.DataFrame()
        steps_hidden_layer_df = pd.DataFrame()
        learning_rates = [0.01, 0.005, 0.001, 0.0005]
        names = ["Learning Rate = 0.01", "Learning Rate = 0.005", "Learning Rate = 0.001", "Learning Rate = 0.0005"]
        for name, learning_rate in zip(names, learning_rates):
            start = time.time()
            env = gym.make('LunarLander-v2')
            # ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
            ddqn = DDQN(9, env.action_space.n, hidden_layer_sizes=[64], learning_rate=learning_rate, gamma=0.99, training_times_per_weight_copy=50)

            rewards = []
            steps_to_finish = []
            for i in range(501):
                #random_action_rate = 1/np.sqrt(i+1)
                reward, steps = run_episode(env, ddqn, random_action_rate, (i%20==0))
                rewards.append(reward)
                steps_to_finish.append(steps)
                hidden_layer_df.loc[i, name] = reward
                steps_hidden_layer_df.loc[i, name] = steps
                random_action_rate *= random_action_decay_rate
                if i % 20 == 0:
                    print("\tEpisode:", i, "\tReward:", reward, "\tSteps:", steps)

            print(name, ":", (time.time()-start))
        print("\n\n\n\n")


        hidden_layer_df.plot()
        plt.title("Learning Rate Reward per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig(("./Figures/Learning Rate Reward vs Episode " + str(trial)))
        plt.close()

        steps_hidden_layer_df.plot()
        plt.title("Learning Rate Steps per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig("./Figures/Learning Rate vs Episode " + str(trial))
        plt.close()

        hidden_layer_df.to_csv("./Data/Learning Rate Experiment " + str(trial) + ".csv")
        steps_hidden_layer_df.to_csv("./Data/Learning Rate Steps " + str(trial) + ".csv")

def buffer_size_experiment():
    sns.set()
    random_action_rate = 0.6
    for trial in range(3, 5):
        random_action_decay_rate = 0.99
        hidden_layer_df = pd.DataFrame()
        steps_hidden_layer_df = pd.DataFrame()
        buffer_sizes = [60000, 80000, 100000, 120000]
        names = ["Buffer Size = 60000", "Buffer Size = 80000", "Buffer Size = 100000", "Buffer Size = 120000"]
        for name, buffer_size in zip(names, buffer_sizes):
            start = time.time()
            env = gym.make('LunarLander-v2')
            # ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
            ddqn = DDQN(9, env.action_space.n, hidden_layer_sizes=[64], experience_buffer_size=buffer_size, gamma=0.99, training_times_per_weight_copy=50)

            rewards = []
            steps_to_finish = []
            for i in range(501):
                #random_action_rate = 1/np.sqrt(i+1)
                reward, steps = run_episode(env, ddqn, random_action_rate, (i%20==0))
                rewards.append(reward)
                steps_to_finish.append(steps)
                hidden_layer_df.loc[i, name] = reward
                steps_hidden_layer_df.loc[i, name] = steps
                random_action_rate *= random_action_decay_rate
                if i % 20 == 0:
                    print("\tEpisode:", i, "\tReward:", reward, "\tSteps:", steps)

            print(name, ":", (time.time()-start))



        hidden_layer_df.plot()
        plt.title("Buffer Size Reward per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig(("./Figures/Buffer Size Reward vs Episode " + str(trial)))
        plt.close()

        steps_hidden_layer_df.plot()
        plt.title("Buffer Size Steps per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig("./Figures/Buffer Size vs Episode " + str(trial))
        plt.close()

        hidden_layer_df.to_csv("./Data/Buffer Size Experiment " + str(trial) + ".csv")
        steps_hidden_layer_df.to_csv("./Data/Buffer Size Steps " + str(trial) + ".csv")


def training_times_before_weight_copy_experiment():
    sns.set()
    random_action_rate = 0.6
    for trial in range(1, 5):
        random_action_decay_rate = 0.99
        hidden_layer_df = pd.DataFrame()
        steps_hidden_layer_df = pd.DataFrame()
        training_times_before_weight_copys = [50, 100, 1000, 10000]
        names = ["Training Times Before Weight Copy = 50", "Training Times Before Weight Copy = 100", "Training Times Before Weight Copy = 1000", "Training Times Before Weight Copy = 10000"]
        for name, training_times_before_weight_copy in zip(names, training_times_before_weight_copys):
            start = time.time()
            env = gym.make('LunarLander-v2')
            # ddqn = DDQN(len(env.observation_space.sample()), env.action_space.n)
            ddqn = DDQN(9, env.action_space.n, hidden_layer_sizes=[64], gamma=0.99, training_times_per_weight_copy=training_times_before_weight_copy)

            rewards = []
            steps_to_finish = []
            for i in range(501):
                #random_action_rate = 1/np.sqrt(i+1)
                reward, steps = run_episode(env, ddqn, random_action_rate, (i%20==0))
                rewards.append(reward)
                steps_to_finish.append(steps)
                hidden_layer_df.loc[i, name] = reward
                steps_hidden_layer_df.loc[i, name] = steps
                random_action_rate *= random_action_decay_rate
                if i % 20 == 0:
                    print("\tEpisode:", i, "\tReward:", reward, "\tSteps:", steps)

            print(name, ":", (time.time()-start))



        hidden_layer_df.plot()
        plt.title("Training Times per Copy Reward per Episode")
        plt.ylabel("Episode Reward")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig(("./Figures/Training Times per Copy Reward vs Episode " + str(trial)))
        plt.close()

        steps_hidden_layer_df.plot()
        plt.title("Training Times per Copy Steps per Episode")
        plt.ylabel("Steps")
        plt.xlabel("Episode")
        plt.tight_layout()
        plt.savefig("./Figures/Training Times per Copy vs Episode " + str(trial))
        plt.close()

        hidden_layer_df.to_csv("./Data/Training Times per Copy Experiment " + str(trial) + ".csv")
        steps_hidden_layer_df.to_csv("./Data/Training Times per Copy Steps " + str(trial) + ".csv")



if __name__ == "__main__":
    training_times_before_weight_copy_experiment()
