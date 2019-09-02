import time
# import matplotlib.pyplot as plt
# import sys
import gym
import numpy as np
import tensorflow as tf
from numpy.random import choice
from tensorflow.keras import layers

UP_ACTION = 2
DOWN_ACTION = 5
STILL_ACTION = 1


# RGB to grayscale
def rgb_to_gray(img):
    return np.dot(img, [0.299, 0.587, 0.144])


# Process image
def preprocess(input_image):
    gray = rgb_to_gray(input_image)

    # plt.imshow(gray, cmap='gray')
    # plt.show()

    gray = gray[33:193, :]
    gray = gray[::4, ::2]

    # plt.imshow(gray, cmap='gray')
    # plt.show()

    return gray / 255


print(tf.__version__)

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), strides=1, activation=tf.keras.activations.relu, input_shape=(40, 80, 2)),
    layers.Conv2D(64, (3, 3), strides=1, activation=tf.keras.activations.relu),
    layers.Conv2D(64, (3, 3), strides=1, activation=tf.keras.activations.relu),
    layers.Flatten(),
    layers.Dense(128, activation=tf.keras.activations.relu),
    layers.Dense(3, activation=tf.keras.activations.softmax)
])
# try:
    # model.load_weights("weights.h5")
# except:
#     print("Unable to load weights")
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

env = gym.make("Pong-v0")

observation = env.reset()

prev_inp = preprocess(observation)

x_train, y_train, rewards = [], [], []

score = []

episode_nb = 0
gamma = 0.99
epsilon = 0
epsilon_dec = 0.995
epsilon_min = 0
nb_sample = 0
rewards_sum = 0
min_batch_size = 20000


def discount_rewards(rewards, gamma):
    accum = 0
    for i in reversed(range(len(rewards))):
        if abs(rewards[i]) < 0.5:
            rewards[i] = accum
        accum = rewards[i] * gamma
    mean = np.mean(rewards)
    stddev = np.std(rewards)
    for i in range(len(rewards)):
        rewards[i] = (rewards[i] - mean) / stddev
    rewards = np.array(rewards)
    return rewards


while True:
    # time.sleep(0.01)

    # if episode_nb % 5 == 0:
    # time.sleep(0.001)
    env.render()

    cur_inp = preprocess(observation)

    # x = cur_inp - prev_inp

    x = np.expand_dims(cur_inp, axis=2)

    x = np.array((cur_inp.T, prev_inp.T)).T

    prev_inp = cur_inp

    # plt.imshow(x[:, :, 0], cmap='Blues')
    # plt.show()
    # plt.close()

    pred = model.predict(np.expand_dims(x, axis=0))  # if np.random.uniform() > epsilon else 0.5

    # print(pred)

    if np.random.uniform() > epsilon:
        y = choice(a=[0, 1, 2], p=pred[0])
    else:
        y = choice(a=[0, 1, 2])

    if y == 0:
        action = UP_ACTION
    elif y == 1:
        action = DOWN_ACTION
    else:
        action = STILL_ACTION

    # y = 1 if action == UP_ACTION else 0

    x_train.append(x)
    y_train.append(y)

    observation, reward, done, info = env.step(action)

    rewards.append(reward)

    rewards_sum += reward

    nb_sample += 1

    if done:
        episode_nb += 1
        print('Episode : ', episode_nb, ' | Score : ', rewards_sum)
        score.append(rewards_sum)
        rewards_sum = 0
        # plt.plot(score)
        # plt.show(block=False)
        # plt.close('all')
        if nb_sample >= min_batch_size:
            model.fit(x=np.array(x_train), y=np.vstack(tf.keras.utils.to_categorical(y_train)), verbose=1,
                      sample_weight=discount_rewards(rewards, gamma))
            model.save_weights('weights.h5')
            x_train, y_train, rewards = [], [], []
            if epsilon * epsilon_dec > epsilon_min:
                epsilon = epsilon * epsilon_dec
            else:
                epsilon = epsilon_min
            nb_sample = 0
        observation = env.reset()
        prev_input = preprocess(observation)

