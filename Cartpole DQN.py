import gym
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

class Layer:
    def __init__(self, inputs, outputs, f=tf.nn.tanh):
        self.weights = tf.Variable(tf.random_normal(shape=(inputs, outputs)))
        self.params = [self.weights]
        self.b = tf.Variable(np.zeros(outputs).astype(np.float32))
        self.params.append(self.b)
        self.f = f

    def forward(self, X):
        return self.f(tf.matmul(X, self.weights) + self.b)

class DQN:
    def __init__(self, inputs, outputs, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_size=32):
        self.num_inputs = inputs
        self.num_outputs = outputs
        self.layers = []
        ins = inputs
        for size in hidden_layer_sizes:
            self.layers.append(Layer(ins, size))
            ins = size
        self.layers.append(Layer(ins, outputs, lambda x: x))

        self.params = []
        for layer in self.layers:
            self.params += layer.params

        self.X = tf.placeholder(tf.float32, shape=(None, inputs), name='X')
        self.G = tf.placeholder(tf.float32, shape=(None,), name='G')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_pred = Z
        self.predict_op = Y_pred
        selected_action_values = tf.reduce_sum(Y_pred*tf.one_hot(self.actions, outputs), reduction_indices=[1])

        cost = tf.reduce_sum(tf.square(self.G - selected_action_values))
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(cost)

        self.experience = []
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_size = batch_size
        self.gamma = gamma

    def set_session(self, session):
        self.session = session

    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual = self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})

    def train(self, target_network):
        if len(self.experience) < self.min_experiences:
            return

        indexes = np.random.choice(len(self.experience), size=self.batch_size, replace=False)
        states = [self.experience[i][0] for i in indexes]
        actions = [self.experience[i][1] for i in indexes]
        rewards = [self.experience[i][2] for i in indexes]
        next_states = [self.experience[i][3] for i in indexes]
        done = [self.experience[i][4] for i in indexes]
        next_Q = np.max(target_network.predict(next_states), axis=1)
        targets = [r + self.gamma*nextq if not done else r for r, nextq, done in zip(rewards, next_Q, done)]

        self.session.run(self.train_op, feed_dict={self.X: states, self.G: targets, self.actions: actions})

    def add_experience(self, s, a, r, s2, done):
        if len(self.experience) >= self.max_experiences:
            self.experience.pop(0)
        self.experience.append((s, a, r, s2, done))

    def get_action(self, x, random_action_rate):
        if np.random.random() < random_action_rate:
            return np.random.choice(self.num_outputs)
        else:
            return np.argmax(self.predict(np.atleast_2d(x))[0])

def play_one(env, model, target_model, random_action_rate, gamma, copy_period):
    state = env.reset()
    done = False
    total_reward = 0
    iters = 0
    while not done and iters < 2000:
        action = model.get_action(state, random_action_rate)
        last_state = state
        state, reward, done, info = env.step(action)

        total_reward += reward

        # if done:
        #     reward = -200

        model.add_experience(last_state, action, reward, state, done)
        model.train(target_model)

        iters += 1

        if iters % copy_period == 0:
            target_model.copy_from(model)

    return total_reward

def main():
    env = gym.make('CartPole-v0')
    gamma = 0.99
    copy_period = 50

    inputs = len(env.observation_space.sample())
    outputs = env.action_space.n
    sizes = [200, 200]
    model = DQN(inputs, outputs, sizes, gamma)
    target_model = DQN(inputs, outputs, sizes, gamma)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    model.set_session(session)
    target_model.set_session(session)


    N = 500
    total_rewards = np.empty(N)
    costs = np.empty(N)
    for n in range(N):
        eps = 1/np.sqrt(n+1)
        total_reward = play_one(env, model, target_model, eps, gamma, copy_period)
        total_rewards[n] = total_reward
        if n%100 == 0:
            print("episode:", n, "\ttotal reward:", total_reward)

    print("Average reward for last 100 episodes:", total_rewards[-100:].mean())

    plt.plot(total_rewards)
    plt.title("Rewards")
    plt.show()

if __name__ == '__main__':
    main()

