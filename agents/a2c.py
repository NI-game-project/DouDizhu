import tensorflow as tf     
import keras 
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

import random
from collections import namedtuple

from envs.utils import remove_illegal

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])


class Actor_Critic():
    
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=20000,
                 batch_size=32,
                 action_num=2,
                 state_shape=None,
                 train_every=256,
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
        self.state_shape = (6,5,15) #516
        self.replay_memory_size = replay_memory_size
        

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay scheduler
        # TODO: Doesnt really matter, but remove it for A2C
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.memory = Memory(self.replay_memory_size, self.batch_size)

        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)

        self.critic = self.create_critic(1, self.state_shape)
        self.critic.compile(optimizer=self.optimizer, loss='mse')
        self.actor = self.create_actor(self.action_num, self.state_shape)
        self.actor.compile(loss='categorical_crossentropy', optimizer=self.optimizer)

        self.history_actor = 0
        self.history_critic = 0
        
    def feed(self, ts):

        (state, action, reward, next_state, done) = tuple(ts)
        self.total_t += 1
        
        if not done:
            self.feed_memory(state['obs'], action, reward, next_state['obs'], done)
        
        else:
            self.feed_memory(state['obs'], action, reward, next_state['obs'], done)
            self.train()
            self.memory.purge_memory()

    def step(self, state):

        A = self.predict(state['obs'].astype(np.float32))
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        
        return action

    def eval_step(self, state):

        prediction = self.actor(np.expand_dims(state['obs'].astype(np.float32), 0))[0]
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        
        return best_action, probs

    def predict(self, state):

        #epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        #A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        prediction = self.actor(np.expand_dims(state.astype(np.float32),0))[0].numpy()
        best_action = np.argmax(prediction)
        prediction[best_action] += 1#(1.0 - epsilon)
        
        return prediction 
        
    def train(self):
        
        states, actions, rewards, _, _ = self.memory.sample()
        values = self.critic(states.astype(np.float32))
        predictions = self.actor(states.astype(np.float32))
        predictions = np.argmax(predictions, axis=1)
        action_one_hot = tf.one_hot(actions, self.action_num, on_value=1,off_value=0)

        discounted_rewards = self.discounted_rewards(rewards)

        advantages = discounted_rewards - tf.reshape(values,-1)
        advantages = np.reshape(advantages, (-1,1))
        
        history_actor = self.actor.fit(states, action_one_hot, sample_weight = advantages, batch_size=self.batch_size, verbose=0,epochs = 10)
        history_critic = self.critic.fit(states, discounted_rewards, batch_size=self.batch_size, verbose = 0, epochs = 10)

        self.history_actor = np.mean(history_actor.history['loss'])
        self.history_critic = np.mean(history_critic.history['loss'])
        self.actions = actions
        self.predictions = predictions

    def discounted_rewards(self, reward):

        gamma = 0.993    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward, dtype='float64')
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        
        #if tf.reduce_sum(reward) != 0:
        #    discounted_r -= np.mean(discounted_r) # normalizing the result
        #    discounted_r /= np.std(discounted_r)
        return discounted_r


    def feed_memory(self, state, action, reward, next_state, done):

        self.memory.save(state, action, reward, next_state, done)

    
    def create_actor(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(1024,activation='relu')(x)
        output = Dense(action_num, activation='softmax')(x)
        network = keras.Model(inputs = input_x, outputs=output)
    
        return network
        
    def create_critic(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        x = Dense(1024,activation='relu')(x)
        output = Dense(action_num)(x)
        network = keras.Model(inputs = input_x, outputs=output)
        
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
        
        samples = self.memory

        return map(np.array, zip(*samples))

    def purge_memory(self):
        
        self.memory = []

