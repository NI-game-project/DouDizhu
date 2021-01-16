''' DQN agent

The code is derived from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/dqn.py

Copyright (c) 2019 DATA Lab at Texas A&M University
Copyright (c) 2016 Denny Britz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''



'''

This is a modified agent by Georg which uses tensorflow 2.3.0, keras 2.4.3, numpy 1.18.5 

'''

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

import keras
from keras.layers import Dense, Input, Flatten, Dropout
from keras.optimizers import Adam

from envs.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class DQNAgent(object):


    # TODO: I dont know what values work here Georg
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 action_num=None,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.0001):

        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.learning_rate = learning_rate
        self.state_shape = (516)


        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.q_estimator = self.create_model(action_num= self.action_num, learning_rate=self.learning_rate, state_shape=self.state_shape)
        self.target_estimator = self.create_model(action_num= self.action_num, learning_rate=self.learning_rate, state_shape=self.state_shape)

        self.memory = Memory(replay_memory_size, batch_size)

    def feed(self, ts):

        (state, action, reward, next_state, done) = tuple(ts)
        self.feed_memory(state['obs'], action, reward, next_state['obs'], done)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size
        if tmp>=0 and tmp%self.train_every == 0:
            self.train()

    def step(self, state):

        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
 
        return action

    def eval_step(self, state):

        q_values = self.q_estimator(np.expand_dims(state['obs'], 0))[0]
        probs = remove_illegal(np.exp(q_values), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

    def predict(self, state):

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        q_values = self.q_estimator(np.expand_dims(state, 0))[0]
        best_action = np.argmax(q_values)
        A[best_action] += (1.0 - epsilon)
        return A

    def train(self):

        # first implementation of a double dqn network refer to https://keras.io/examples/rl/deep_q_network_breakout/     
        state_batch, _, reward_batch, next_state_batch, done_batch = self.memory.sample()
        
        # Calculate q values and targets (Double DQN)
        # first predict the q_values with the q_estimator and choose the best action
        q_values_next = self.q_estimator(next_state_batch).numpy()
        best_actions = np.argmax(q_values_next, axis=1)

        #than predict the same with the target_estimator
        q_values_next_target = self.target_estimator(next_state_batch).numpy()

        # in order to get the target, add the predicted next q_values (reward) times a discount factor to the reward
        # the invert done batch states, that at the end of the episode, no future state is taken into account
        target_batch = reward_batch + np.invert(done_batch).astype(np.float32) * self.discount_factor * q_values_next_target[np.arange(self.batch_size), best_actions]
        state_batch = np.array(state_batch)

        self.q_estimator.fit(state_batch, target_batch, batch_size= self.batch_size, verbose=0)       

        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            
            self.target_estimator.set_weights(self.q_estimator.get_weights())
            print("\nINFO - Copied model parameters to target network.")

        self.train_t += 1

    def feed_memory(self, state, action, reward, next_state, done):
  
        self.memory.save(state, action, reward, next_state, done)


    def create_model(self, action_num, learning_rate, state_shape):
        
        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        output = Dense(action_num)(x)
        network = keras.Model(inputs = input_x, outputs=output)
        network.compile(loss='mse', optimizer=Adam(lr=learning_rate))
        
        return network
        
class Memory(object):


    def __init__(self, memory_size, batch_size):
 
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):

        samples = random.sample(self.memory, self.batch_size)

        return map(np.array, zip(*samples))

