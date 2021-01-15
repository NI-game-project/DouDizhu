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

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'done', 'probs', 'value'])

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
                 action_num=131,
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
        self.model = Network(self.action_num, self.hidden)


    def feed(self, ts):
        '''
        Feeds data to RL agents memory and trains after specified stepzize self.train_every

        Args:
            ts (list): A list of 5 elements that represent the transition.
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        self.total_t += 1
        #print('print(state)',state['obs'], self.total_t)

        old_prob, old_value = self.model(np.expand_dims(state['obs'], 0))

        if not self.total_t % self.train_every == 0:
            
            
            # prediction[0] is the output of the policy/actor network for given state
            # prediction[1] is the output of the value/critic network
            new_memory = np.array([state, action, reward, done, old_prob.numpy(), old_value.numpy()], dtype=object)
            self.feed_memory(new_memory)
        
        else:
            #print(self.total_t % self.train_every)
            new_memory = np.array([state, action, reward, done, old_prob.numpy(), old_value.numpy()], dtype=object)
            self.feed_memory(new_memory)
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
        pred, value = self.model(np.expand_dims(state['obs'],0))
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
        prediction = self.model(np.expand_dims(state['obs'], 0))[0]
        prediction = prediction[0].numpy()
        probs = remove_illegal(np.exp(prediction), state['legal_actions'])
        best_action = np.argmax(probs)
        #print(np.exp(prediction))
        return best_action, probs

        
    def train(self):
        
        for _ in range(self.epochs):

            states_x, actions, rewards, dones, old_probs, old_values = self.memory.sample()

            states = []
            for i in states_x:
                b = i['obs']
                states.append(b)

            states = np.stack(states)
            #states = states[:,'obs']
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
            '''
            penalty = np.zeros(batch_size)
            for i in range(batch_size):
                if best_actions[i] not in states_legal[i]:
                    penalty[i] += 1

            self.penalty = penalty
            
            self.best_actions = best_actions
            self.actions = actions
            '''
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
                '''
                states = states[batch]
                states = states[np.newaxis,:]
                old_probs = old_probs[batch]
                actions = actions[batch]
                advantages = advantages[batch]
                old_values = old_values[batch]
                '''
                with tf.GradientTape() as tape:
                    
                    
                    prob, value = self.model(states[batch])
                    

                    old_prob = old_probs[batch]
                    old_value = old_values[batch]
                    advantage = advantages[batch]
                    action = actions[batch]
                    
                    #probs = probs[0]
                    #values = values[0]
                    #action_one_hot = tf.one_hot(actions, self.action_num, on_value=1,off_value=0)
                    #advantages = rewards - tf.reshape(values,-1) - penalty
                    #prob = tf.gather(tf.reshape(predictions, [-1]), actions,axis=0)

                    loss_clipping = 0.2
                    entropy_coeff = 0.01
                    
                    a0 = prob - tf.reduce_max(prob, axis=-1, keepdims=True)
                    ea0 = tf.exp(a0)
                    z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
                    p0 = ea0 / z0
                    entropy = tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)
                    mean_entropy = tf.reduce_mean(entropy) 
                    entropy_loss = - mean_entropy * entropy_coeff 

                    #ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
                    #prob = y_pred_actor * action_one_hot
                    #old_prob = action_one_hot * predictions

                    # The problem is that the actions taken arent the ones it predicted. so I probably should fit for the taken actions

                    prob = tfp.distributions.Categorical(probs=prob)
                    prob = prob.log_prob(action)

                    old_prob = tfp.distributions.Categorical(probs=old_prob)
                    old_prob = old_prob.log_prob(action)
                    

                    r = tf.math.exp(prob - old_prob)
                    p1 = r * advantage
                    p2 = K.clip(r, min_value=1-loss_clipping, max_value=1+loss_clipping) * advantage
                    
                    policy_loss = - K.mean(K.minimum(p1,p2))

                    combined = advantage + old_value
                    value_loss = K.mean(tf.losses.mse(combined, value))
                    value_loss = tf.cast(value_loss, 'float32')           
                    loss = policy_loss + entropy_loss + 0.5 * value_loss 
                    #print(policy_loss, 'policy \n', entropy_loss, 'entropy\n', value_loss, 'value\n')
                    grads = tape.gradient(loss, self.model.trainable_variables)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))


    def feed_memory(self, new_memory):
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

        self.memory.save(new_memory)

    

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

class Memory(object):

    def __init__(self, memory_size, batch_size):

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []
    
    def save(self, new_memory):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        state, action, reward, done, probs, values = new_memory

        
        #transition = np.array([state, state_legal, action, reward, next_state, done, probs, values], dtype='object')
        transition = Transition(state, action, reward, done, probs[0], values[0])
        self.memory.append(transition)
    
    def sample(self):

        samples = self.memory

        return map(np.array, zip(*samples))

    def purge_memory(self):

        self.memory = []

