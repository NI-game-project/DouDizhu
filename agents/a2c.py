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


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])


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
                 initial_learning_rate=5e-4, 
                 decay_steps=2000, 
                 decay_rate=0.95):
        
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.train_every = train_every
        self.state_shape = (6,5,15) #516
        self.replay_memory_size = replay_memory_size
        

        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0

        # The epsilon decay schedulers
        # TODO: Doesnt really matter, but remove it for A2C
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        self.memory = Memory(self.replay_memory_size, self.batch_size)

        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.initial_learning_rate, self.decay_steps, self.decay_rate)
        lr_schedule_v = tf.keras.optimizers.schedules.ExponentialDecay(1e-5, self.decay_steps, self.decay_rate)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.optimizer_v = tf.keras.optimizers.Adam(learning_rate=lr_schedule_v)
        self.critic = self.create_critic(1, self.state_shape)
        self.critic.compile(optimizer=self.optimizer, loss='mse')
        print(self.critic.summary())

        self.actor = self.create_actor(self.action_num, self.state_shape)
        self.actor.compile(loss='mse', optimizer=self.optimizer)
        print(self.actor.summary())
        self.history_actor = 0
        self.history_critic = 0
        
    def feed(self, ts):

        (state, action, reward, next_state, done) = tuple(ts)
        self.total_t += 1
        
        if not done:
            self.feed_memory(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'])
        
        else:
            self.feed_memory(state['obs'], action, reward, next_state['obs'], done, state['legal_actions'] )
            self.train()
            self.memory.purge_memory()

    def step(self, state):

        A = self.predict(state['obs'].astype(np.float32))
        #print(A,'first')
        A = remove_illegal(A, state['legal_actions'])
        #print(A,'second')
        action = np.random.choice(np.arange(len(A)), p=A)
        
        return action

    def eval_step(self, state):

        prediction = self.actor(np.expand_dims(state['obs'].astype(np.float32), 0))[0]
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        
        return best_action, probs

    def predict(self, state):

        prediction = self.actor(np.expand_dims(state.astype(np.float32),0))[0].numpy()
        prediction += 1e-5

        return prediction 
        
    def train(self):
        
        states, actions, rewards, next_state, done, legal_actions = self.memory.sample()
        
        best_action = tf.argmax(self.actor(states.astype(np.float32)), axis=1)

        with tf.GradientTape() as tape_v:

            penalty = np.zeros(len(actions))
            for i in range(len(actions)):
                x = best_action[i]
                if x in legal_actions[i]:
                    penalty[i] -= 0.2 # before 0.01
                if x == 308:
                    penalty[i] += 1

            values = self.critic(states.astype(np.float32))
            rewards = self.discounted_rewards(rewards) - penalty.T
            values = tf.reshape(values, -1)
            #mse = tf.keras.losses.MeanSquaredError()
            #value_loss = mse(values, rewards)
            #loss_f = tf.keras.losses.MeanSquaredLogarithmicError()
            #value_loss = loss_f(rewards, values)
            value_loss = tf.math.reduce_mean(tf.math.square(values-rewards.T))
            value_loss = value_loss/len(actions)
            grads = tape_v.gradient(value_loss, self.critic.trainable_variables)
            self.optimizer_v.apply_gradients(zip(grads,self.critic.trainable_variables))
        
        values_next = self.critic(next_state.astype(np.float32))
        #print(values.numpy())
        
        #print(predictions.numpy())
        #predictions = np.argmax(predictions, axis=1)
        #action_one_hot = tf.one_hot(actions, self.action_num, on_value=1,off_value=0)


        
        gamma = 0.9
        

        # This is only one step look ahead. Probably many step look a head is better
        advantages = rewards - tf.reshape(values,-1) + gamma*tf.reshape(values_next, -1)*np.invert(done).astype(np.float32)# -penalty
        '''
        batch_size = len(actions)
        advantages = np.zeros(batch_size)
        for i in range(batch_size-1):
            discount = self.discount_factor # check here the decay rate (compare to the paper)
            a = 0
            for j in range(i, batch_size-1):
                a += discount * (rewards[j] + np.invert(done[j]).astype(np.float32)*gamma*values_next[j]) - values[j]
                discount *= discount
            advantages[i] = a 
        advantages = np.reshape(advantages, (-1,1))
        #advantages = (advantages - advantages.mean() )/(advantages.std()+1e-8) 
        '''
        with tf.GradientTape() as tape:

            loss_clipping = 0.2
            probs = self.actor(states.astype(np.float32))
            probs += 1e-9

            prob = tfp.distributions.Categorical(probs=probs)
            prob = prob.log_prob(actions)

            old_prob = tfp.distributions.Categorical(probs=probs)
            old_prob = old_prob.log_prob(actions)

            r = tf.math.exp(prob - old_prob)
            p1 = r * advantages
            p2 = K.clip(r, min_value=1-loss_clipping, max_value=1+loss_clipping) * advantages
            
            policy_loss = - K.mean(K.minimum(p1,p2))

            #combined = advantage + old_value            



            #best_action = tf.argmax(prob, axis=1)

            #penalty = tf.Variable(tf.zeros(len(actions)),shape=best_action.shape)

            
            
            entropy_coeff = 1
            a0 = prob - tf.reduce_max(prob, axis=-1, keepdims=True)
            ea0 = tf.exp(a0)
            z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
            p0 = ea0 / z0
            entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
            mean_entropy = tf.reduce_mean(entropy) 
            entropy_loss = - mean_entropy * entropy_coeff 
            

            prob_p = tfp.distributions.Categorical(probs=probs)

            action_log_probs =  prob_p.log_prob(actions + 1e-9)
            #fehler = prob_p.log_prob(best_action + 1e-9)
            best_action = tf.argmax(self.actor(states.astype(np.float32)), axis=1)

            q = prob_p.log_prob(best_action)
            actor_loss_2 = q * 0.01# this worked0.001
            #for i in range(len(actions)):
            #    if actions[i] == 308:
            #        actions[i] == np.random.randint(0,307)
            #action_one_hot = tf.one_hot(actions, self.action_num, on_value=1,off_value=0).numpy().astype(np.float32)
            
            #action_log_probs2 = - tf.nn.softmax_cross_entropy_with_logits(action_one_hot,prob)

            #action_log_probs = tf.math.log(actions_prob)
            actor_loss = action_log_probs * advantages
            actor_loss = - tf.reshape(actor_loss,-1) #+ penalty
            #actor_loss = - tf.tensordot(action_log_probs, advantages,1)
            actor_loss = - tf.reduce_mean(actor_loss - actor_loss_2) 
            actor_loss = actor_loss/len(actions) + entropy_loss #+ penalty
            grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))
            predictions = tf.experimental.numpy.amax(probs, axis =1)#prob.numpy()


        
        #advantages = advantages**2
        
        #history_actor = self.actor.fit(states, action_one_hot, sample_weight = advantages, batch_size=self.batch_size, verbose=0,epochs = 10)
        
        
        history_actor = actor_loss
        #history_critic = self.critic.fit(states, rewards, batch_size=len(actions), verbose = 0, epochs = 1)

        self.history_actor = history_actor #np.mean(history_actor.history['loss'])
        self.history_critic = q# value_loss#np.mean(history_critic.history['loss'])
        self.actions = actions#prob.numpy()[:][:5]
        self.predictions  = best_action #f.math.exp(action_log_probs)#predictions.argsort()[:][:5]


    def discounted_rewards(self, reward):

        gamma = 0.9  # discount rate
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

    def discount_with_dones(self, rewards, dones, gamma):
        discounted = []
        r = 0
        for reward, done in zip(rewards[::-1], dones[::-1]):
            r = reward + gamma*r*(1.-done) # fixed off by one bug
            discounted.append(r)
        return discounted[::-1]

    def feed_memory(self, state, action, reward, next_state, done, legal_actions):

        self.memory.save(state, action, reward, next_state, done, legal_actions)

    
    def create_actor(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        #x = Dense(1024,activation='relu')(x)
        output = Dense(action_num, activation='softmax')(x)
        network = keras.Model(inputs = input_x, outputs=output)
    
        return network
        
    def create_critic(self, action_num, state_shape):

        input_x = Input(state_shape)
        x = Flatten()(input_x)
        x = Dense(512,activation='relu')(x)
        x = Dense(512,activation='relu')(x)
        #x = Dense(1024,activation='relu')(x)
        output = Dense(action_num)(x)
        network = keras.Model(inputs = input_x, outputs=output)
        
        return network


class Memory(object):

    def __init__(self, memory_size, batch_size):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
    
    def save(self, state, action, reward, next_state, done, legal_actions):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        
        transition = Transition(state, action, reward, next_state, done, legal_actions)
        self.memory.append(transition)
    
    def sample(self):
        
        samples = self.memory

        return map(np.array, zip(*samples))

    def purge_memory(self):
        
        self.memory = []

