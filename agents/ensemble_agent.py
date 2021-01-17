'''
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

This is a modified agent by Georg which uses tensorflow 2.4.0, keras 2.4.3, numpy 1.18.5 

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

        self.gamma = 0.97
        self.hidden = 512
        self.q_estimator = Network(action_num= self.action_num, hidden=self.state_shape)
        self.target_estimator = Network(action_num= self.action_num, hidden=self.state_shape)

        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.q_estimator.compile(self.optimizer, loss='mse')
        self.target_estimator.compile(self.optimizer)

        self.memory = Memory(replay_memory_size, batch_size)

        if i == 0:
            with h5py.File('models/genetic/a2c_a2_0.392.h5', 'r') as hf:
                elite_workers = hf['models/genetic/a2c_a2_0.392.h5'][:]

        #agent.actor = keras.models.load_model('models/a2c/actor_d3.h5')
        #agent.critic = keras.models.load_model('models/a2c/critic_d3.h5')

        # for dqn agent

        #agent.q_estimator = keras.models.load_model('models/dqn/q_estimator_v1.h5')
        #agent.target_estimator = keras.models.load_model('models/dqn/target_estimator_v1.h5')
            
    def eval_step(self, state):

        probs = self.ensemble(np.expand_dims(state['obs'], 0))
        probs = remove_illegal(np.exp(probs), state['legal_actions'])
        best_action = np.argmax(probs)
        return best_action, probs

    def ensemble(self, state):

        for network in range(self.ensembles):

            self.set_weights(network)
            prob = self.model(state)[0]
            prediction += prob
        
        return prediction / prediction.mean()
    

    def set_weights(self, network):
        
        
        pass



        