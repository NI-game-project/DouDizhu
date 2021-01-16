import tensorflow as tf     
import keras 
from keras.layers import Dense, Input, Flatten, BatchNormalization
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

import random
from collections import namedtuple

from envs.utils import remove_illegal

import tensorflow_probability as tfp

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'old_prob', 'old_value',  'done'])


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
        self.state_shape = (516) #(6,5,15)
        self.replay_memory_size = replay_memory_size

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0
        self.epochs = 10
        self.hidden = 512
        self.gamma = 0.97
        # The epsilon decay scheduler
        # TODO: Doesnt really matter, but remove it for A2C
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.memory = Memory(self.replay_memory_size, self.batch_size)
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        #self.model = Network(self.action_num, self.hidden)

        #self.model.compile(self.optimizer,loss='mse')

        self.critic = self.create_critic(1, self.state_shape)
        self.critic.compile(optimizer=self.optimizer, loss='mse')
        self.actor = self.create_actor(self.action_num, self.state_shape)
        self.actor.compile(loss='categorical_crossentropy', optimizer=self.optimizer)
        
    def feed(self, ts):

        (state, action, reward, _, done) = tuple(ts)
        self.total_t += 1
        
        old_prob = self.actor(np.expand_dims(state['obs'], 0))[0]
        old_value = self.critic(np.expand_dims(state['obs'], 0))[0]

        state['old_prob'] = old_prob.numpy()
        state['old_value'] = old_value.numpy()

        if not self.total_t % self.train_every == 0:
            self.feed_memory(state['obs'], action, reward, state['old_prob'], state['old_value'], done)
        
        else:
            self.feed_memory(state['obs'], action, reward, state['old_prob'], state['old_value'], done)
            self.train()
            self.memory.purge_memory()

    def step(self, state):

        A = self.predict(state['obs'])
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        
        return action

    def eval_step(self, state):

        prediction = self.actor(np.expand_dims(state['obs'], 0))[0]
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        
        return best_action, probs

    def predict(self, state):

        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        prediction = self.actor(np.expand_dims(state,0))[0]
        best_action = np.argmax(prediction)
        A[best_action] += (1.0 - epsilon)
        
        return A 
        
    def train(self):

        for _ in range(self.epochs):

            states, actions, rewards, old_probs, old_values, dones = self.memory.sample()

            batches = []
            minibatch_size = 16
            batch_size = len(rewards)
            idx_batch = np.arange(batch_size)
            np.random.shuffle(idx_batch)
            idx_mini_batch = np.arange(0,batch_size, minibatch_size)

            for i in idx_mini_batch:
                batches.append(idx_batch[i:i+minibatch_size])

            advantages = np.zeros(batch_size)
            old_values = np.reshape(old_values,(batch_size))
            old_probs = np.reshape(old_probs, (batch_size, -1))

            best_actions = np.argmax(old_probs, axis =1)

            #advantages = advantages-penalty
            #print(best_actions)
            
            for i in range(batch_size-1):
                discount = self.discount_factor # check here the decay rate (compare to the paper)
                a = 0
                for j in range(i, batch_size-1):
                    a += discount * (rewards[j] + np.invert(dones[j]).astype(np.float32)*self.gamma*old_values[j+1]) - old_values[j]
                    discount *= discount
                advantages[i] = a

            
            # maybe the normilazation of the advantages helps improve performance
            advantages = (advantages - advantages.mean())/ advantages.std()
            
            
            for batch in batches:
                
                old_prob = old_probs[batch]
                old_value = old_values[batch]
                advantage = advantages[batch]
                action = actions[batch]
                
                with tf.GradientTape() as tape_v:
                    
                    value = self.critic(states[batch])

                    combined = advantage + old_value
                    value_loss = K.mean(tf.losses.mse(combined, value))
                    value_loss = 0.5 * tf.cast(value_loss, 'float32') 
                    
                    grads = tape_v.gradient(value_loss, self.critic.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

                with tf.GradientTape() as tape_p:

                    prob = self.actor(states[batch])

                    loss_clipping = 0.2
                    entropy_coeff = 0.01
                    
                    a0 = prob - tf.reduce_max(prob, axis=-1, keepdims=True)
                    ea0 = tf.exp(a0)
                    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                    p0 = ea0 / z0
                    entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
                    mean_entropy = tf.reduce_mean(entropy) 
                    entropy_loss = - mean_entropy * entropy_coeff 

                    prob = tfp.distributions.Categorical(probs=prob)
                    prob = prob.log_prob(action)

                    old_prob = tfp.distributions.Categorical(probs=old_prob)
                    old_prob = old_prob.log_prob(action)

                    r = tf.math.exp(prob - old_prob)
                    p1 = r * advantage
                    p2 = K.clip(r, min_value=1-loss_clipping, max_value=1+loss_clipping) * advantage
                    
                    policy_loss = - K.mean(K.minimum(p1,p2))
                              
                    loss = policy_loss + entropy_loss #+ 0.5 * value_loss 
                    #print(policy_loss, 'policy \n', entropy_loss, 'entropy\n', value_loss, 'value\n')
                    grads = tape_p.gradient(loss, self.actor.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))
                    
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


    def feed_memory(self, state, action, reward, old_prob, old_value,done):

        self.memory.save(state, action, reward, old_prob, old_value, done)

    def create_actor(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        output = Dense(action_num, activation='softmax')(x)
        network = keras.Model(inputs = input_x, outputs=output)
    
        return network
        
    def create_critic(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        output = Dense(action_num)(x)
        network = keras.Model(inputs = input_x, outputs=output)
        
        return network

class Network(tf.keras.Model):

    def __init__(self, action_num, hidden):
        super().__init__()

        self.action_num = action_num
        self.hidden = hidden

        self.flatten = Flatten()
        self.dense1 = Dense(self.hidden, activation='relu')
        self.dense_a = Dense(self.hidden, activation='relu')
        self.dense_c = Dense(self.hidden, activation='relu')
        self.actor = Dense(self.action_num, activation='softmax')
        self.critic = Dense(1)
    
    def call(self, inputs):

        x = self.flatten(inputs)
        x = self.dense1(x)
        a = self.dense_a(x)
        c = self.dense_c(x)

        return self.actor(a), self.critic(c)
    
    def call_act(self, inputs):

        x = self.flatten(inputs)
        x = self.dense1(x)
        a = self.dense_a(x)

        return self.actor(a)

class Memory(object):

    def __init__(self, memory_size, batch_size):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
    
    def save(self, state, action, reward, old_prob, old_value, done):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        
        transition = Transition(state, action, reward, old_prob, old_value, done)
        self.memory.append(transition)
    
    def sample(self):
        
        samples = self.memory

        return map(np.array, zip(*samples))

    def purge_memory(self):
        
        self.memory = []

