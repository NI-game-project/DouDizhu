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


# This is given by the RL framework but expanded the legal states 'state_legal' as well as the 'value' of the 
# critic/value network and 'probs' for the calculation of ratio for ppo

Transition = namedtuple('Transition', ['state', 'state_legal', 'action', 'reward', 'next_state', 'done', 'probs', 'value'])

class Actor_Critic():
    '''
    # This is the main class of the Agent according to the framwork of the RL paper. It needs 
    # 1. a feed function for the Memory which is in an seperate class and the training
    # 2. a step function for the interaction with the run function of the enviroment class
    # 3. a eval_step function which interacts with the tournament function from the utils file
    # 4. a train function which does the training
    # 5. a feed_memory function which interacts with the memory class
    '''
    # almost all of these hyperparameters haven't been optimized, so in order to squeeze some percentages in performance, there
    # surely is some space
    def __init__(self,
                 replay_memory_size=20000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.9, # check this value
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay_steps=2000,
                 batch_size=32,
                 action_num=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=None,
                 learning_rate=0.000025):
        
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.epsilon_decay_steps = epsilon_decay_steps
        self.batch_size = batch_size
        self.action_num = action_num
        self.hidden = 512
        self.train_every = train_every
        self.learning_rate = learning_rate
        self.state_shape = (6,5,15)
        self.replay_memory_size = replay_memory_size

        self.best_actions = [1]
        self.penalty = 0
        self.epochs = 5
        self.gamma = 0.9 # check the value in the paper

        # Total timesteps
        self.total_t = 0
        self.train_every = 64
        # Total training step
        self.train_t = 0
        self.actions = [1]

        # TODO: use this epsilon for the decay 
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

        # here the memory as well as the networks are initialized from other classes at the end of this file
        self.memory = Memory(self.replay_memory_size, self.batch_size)
        self.optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.q_eval = Network(self.action_num, self.hidden)
        self.q_next = Network(self.action_num, self.hidden)


    def feed(self, ts, prediction):
        '''
        Feeds data to RL agents memory and trains after specified stepzize self.train_every

        Args:
            ts (list): A list of 5 elements that represent the transition.
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.total_t += 1
        
        if not self.total_t % self.train_every == 0:
            
            # prediction[0] is the output of the policy/actor network for given state
            # prediction[1] is the output of the value/critic network
            self.feed_memory(state['obs'], state['legal_actions'], action, reward, next_state['obs'], done, prediction[0], prediction[1])
        
        else:
            #print(self.total_t % self.train_every)
            self.feed_memory(state['obs'], state['legal_actions'], action, reward, next_state['obs'], done, prediction[0], prediction[1])
            self.train()
            self.memory.purge_memory()

    def step(self, state):
        ''' Returns the action to be taken.

        Args:
            state (dict): The current state
        Returns:
        Returns:
            action (int): An action id
        '''
        # TODO: it still has here the greed epsilon strategy, but this should be compensated by the entropy within the training
        # so delete this later. For now, it just slows down convergence.
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps - 1)]
        A = np.ones(self.action_num, dtype=float) * epsilon / self.action_num
        pred = self.q_eval(np.expand_dims(state['obs'],0))
        prediction = [pred,value]
        probs = prediction[0]
        best_action = np.argmax(probs)
        A[best_action] += (1.0 - epsilon)
        A = remove_illegal(A, state['legal_actions'])
        action = np.random.choice(np.arange(len(A)), p=A)
        
        # Here has been change in the env.py file, because the entire output of the network has to be stored in the trajectory 
        # and not only the action
        # TODO: This might has to be changed for the DQN as well with some None_type or something
        return action

    def eval_step(self, state):
        ''' Use the average policy for evaluation purpose

        Args:
            state (dict): The current state.

        Returns:
            action (int): An action id.
            probs (list): The list of action probabilies
        '''
        # Doesn't need any exploration parameters so just returns actions and their probability
        prediction, _ = self.model(np.expand_dims(state['obs'], 0))
        prediction = tf.reshape(prediction, -1)
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        
        return best_action, probs

        
    def train(self):
        
        #TODO: THIS IS MAYBE NOT RIGHT SO DOUBLE CHECK IT Georg
        # first implementation of a double dqn network refer to https://keras.io/examples/rl/deep_q_network_breakout/
        # it might be wrong, will check within this week

        #state_batch, state_batch_legal, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample()

        states, states_legal, actions, rewards, next_states, dones, old_probs, old_values = self.memory.sample()

        #state_batch = state_batch[0]
        #action_batch = action_batch[0]
        #reward_batch = reward_batch[0]
        #next_state_batch = next_state_batch[0]
        #done_batch = done_batch[0]
        
        #self.batch_size = 1
        # Calculate q values and targets (Double DQN)
        # first predict the q_values with the q_estimator and choose the best action
        q_pred = self.q_eval(states)
        q_next = self.q_next(next_states)

        q_target = q_pred.numpy()
         #q_value = self.q_estimator(np.expand_dims(state_batch, 0))[0]
        #best_action = np.argmax(q_value)
        best_actions = tf.math.argmax(q_next, axis =1)
        
        penalty = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            if best_actions[i] not in states_legal[i]:
                penalty[i] += 0.05



        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + self.gamma * q_next[idx, best_actions[idx]] * np.invert(dones).astype(np.float32)

        self.q_eval.train_on_batch(states, q_target)


        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            
            self.q_next.set_weights(self.q_eval.get_weights())
            print("\nINFO - Copied model parameters to target network.")
            #print(loss)

        self.train_t += 1

    def feed_memory(self, state, states_legal, action, reward, next_state, done, probs, values):
        ''' Function which interacts with the Memory class

        Args:
            state (dict): The current state.
            states_legal (dic): The legal actions.
            action (int): An action id.
            reward(list): Reward for the current state.
            next_state(dic): The next state according to the action. 
            done: Done flag
            probs (list): The list of action probabilies
            values         
            
        '''

        self.memory.save(state, states_legal, action, reward, next_state, done, probs, values)
    

class Network(tf.keras.Model):

    def __init__(self, action_num, hidden):
        super().__init__()

        self.action_num = action_num
        self.hidden = hidden

        self.flatten = Flatten()
        self.dense1 = Dense(self.hidden, activation='relu')
        self.dense2 = Dense(self.hidden, activation='relu')
        self.value = Dense(1)
        self.advantage = Dense(self.action_num)
    
    def call(self, inputs):

        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        v = self.value(x)
        a = self.advantage(x)
        
        return (v + (a - tf.math.reduce_mean(a, axis=1, keepdims=True)))

    def call_advantage(self, inputs):

        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)

        return self.advantage(x)


class Memory(object):

    def __init__(self, memory_size, batch_size):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
    
    def save(self, state, state_legal, action, reward, next_state, done, probs, values):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, state_legal, action, reward, next_state, done, probs, values)
        self.memory.append(transition)
    
    def sample(self):

        samples = self.memory

        return map(np.array, zip(*samples))

    def purge_memory(self):

        self.memory = []

