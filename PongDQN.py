import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

UP_ACTION = 2
DOWN_ACTION = 5
NO_ACTION = 0


class Agent:
    def __init__(self, state_shape, action_nb):
        self.state_shape = state_shape
        self.action_nb = action_nb
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999995
        self.learning_rate = 0.0005
        self.model = self.build_model()

    def build_model(self):
        model = tf.keras.Sequential([
            layers.Conv2D(16, (8, 8), strides=4, activation=tf.keras.activations.relu, input_shape=self.state_shape),
            layers.Conv2D(32, (4, 4), strides=2, activation=tf.keras.activations.relu),
            layers.Flatten(),
            layers.Dense(256, activation=tf.keras.activations.relu),
            layers.Dense(self.action_nb, activation=tf.keras.activations.linear)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                      loss=tf.keras.losses.mean_squared_error)
        model.summary()
        return model

    def make_action(self, state):
        action_code = 0
        if self.epsilon > np.random.rand():
            action_code = np.random.randint(self.action_nb)
        else:
            predict = self.model.predict(np.expand_dims(state, axis=0))
            action_code = np.argmax(predict[0])
        return action_code

    def record(self, state, new_state, action, reward, done):
        self.memory.append([state, new_state, action, reward, done])

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return 0
        batch = random.sample(self.memory, batch_size)
        train_state = []
        train_reward = []
        for state, new_state, action, reward, done in batch:
            target_reward = self.model.predict(np.expand_dims(state, axis=0))[0]
            target_reward[action] = reward
            if not done:
                target_reward[action] += self.gamma * np.amax(self.model.predict(np.expand_dims(new_state, axis=0))[0])
            train_state.append(state)
            train_reward.append(target_reward)
        self.model.fit(np.array(train_state), np.array(train_reward), verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min


def preprocess_input(input_state):
    processed_input = input_state
    processed_input = np.dot(processed_input, [0.299, 0.587, 0.144])
    processed_input = processed_input[33:193, :]
    processed_input = processed_input[::2, ::2]
    processed_input = processed_input / 255.0
    return processed_input


if __name__ == "__main__":
    env = gym.make('Pong-v0')
    agent = Agent((80, 80, 4), 3)
    episode_nb = 0
    observation = env.reset()
    observation = preprocess_input(observation)
    observation = np.full((4, 80, 80), observation)
    state = np.dstack(observation)
    reward_sum = 0
    while True:
        observation[0:2] = observation[1:3]
        env.render()
        action_code = agent.make_action(state)
        if action_code == 0:
            action = NO_ACTION
        if action_code == 1:
            action = UP_ACTION
        if action_code == 2:
            action = DOWN_ACTION
        for frame in range(3):
            new_observation, reward, done, info = env.step(action)
            # env.render()
            reward_sum += reward
        observation[3] = preprocess_input(new_observation)
        new_state = np.dstack(observation)
        agent.record(state, new_state, action_code, reward, done)
        state = new_state
        if done:
            episode_nb += 1
            print("Episode : {} | Avg. Reward : {}".format(episode_nb, reward_sum / episode_nb))
            env.reset()

        agent.train(32)
