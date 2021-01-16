import tensorflow as tf
import numpy as np
import h5py
import matplotlib.pyplot as plt
import random
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from keras.optimizers import Adam, RMSprop

import agents.random_agent as random_agent
import agents.doudizhu_rule_models
import envs.doudizhu
import envs.logger
from envs.utils import set_global_seed, tournament
import agents.dqn

import pandas as pd

class Genetic_Algorithm():

    def __init__(self, population_size=1000, elite_workers_num = 50, evaluation_num = 50, generations = 300, epsilon = 0.001):
        
        self.config = {  'allow_step_back':True,
        'allow_raw_data': True, 
        'record_action': True,
        'seed': 42,
        'single_agent_mode': False,
        'active_player': True}

        self.model_load_path = 'models/genetic/d2_0.489.h5'
        self.model_save_path = 'models/genetic/d3_'
        self.log_dir = 'experiments/genetic/d3'

        self.population_size = population_size
        self.elite_workers_num = elite_workers_num
        self.evaluate_num = evaluation_num
        self.generations = generations
        self.epsilon = epsilon
        
        self.logger = envs.logger.Logger(self.log_dir)
        self.workers = []
        
        self.env = envs.doudizhu.DoudizhuEnv(self.config)
        self.agent = agents.dqn.DQNAgent(action_num=self.env.action_num)
        self.random_agent = agents.random_agent.RandomAgent(action_num=self.env.action_num)
        self.rule_based_agent = agents.doudizhu_rule_models.DouDizhuRuleAgentV1()
        
        
        # Change this to make it generally applicable
        # print(self.agent.q_estimator.summary())
        self.weight_space = 652985 #self.agent.q_estimator.trainable_weights
            
    def set_parameters(self, weights):
        
        last_used = 0
        weights = tf.constant(weights[0], dtype='float32')

        for i in range(len(self.agent.q_estimator.layers)):

            if 'dense' in self.agent.q_estimator.layers[i].name:
                weights_shape = self.agent.q_estimator.layers[i].kernel.shape
                no_of_weights = tf.reduce_prod(weights_shape)
                new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                self.agent.q_estimator.layers[i].kernel = new_weights
                last_used += no_of_weights
                
                if self.agent.q_estimator.layers[i].use_bias:
                    weights_shape = self.agent.q_estimator.layers[i].bias.shape
                    no_of_weights = tf.reduce_prod(weights_shape)
                    new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
                    self.agent.q_estimator.layers[i].bias = new_weights
                    last_used += no_of_weights
        
    def initial_population(self):

        # These are hyperparameters and can be changed
        fan_in = 6400*4
        
        for _ in range(self.population_size):

            # These are hyperparameters and can be changed
            tf.keras.initializers.he_uniform()
            z = np.random.uniform(low=-np.sqrt(6/fan_in), high=np.sqrt(6/fan_in), size=self.weight_space)
            self.workers.append([z])
        
    def evaluate_population(self):

        elite_workers = []
        rewards = []

        for weights in self.workers:
            
            self.set_parameters(weights)
            #self.env.set_agents([self.model, self.random_agent, self.random_agent])
            self.env.set_agents([self.agent, self.rule_based_agent, self.rule_based_agent])
        
            payoff = tournament(self.env, self.evaluate_num)[0]
            rewards.append(payoff)
        
        rewards = np.array(rewards)
        elite_idx = np.argsort(rewards)[self.population_size-self.elite_workers_num:]

        for idx in elite_idx:
            elite_workers.append(self.workers[idx])

        return elite_workers, rewards

    def mutate_population(self, elite_workers):

        self.workers = []

        for i in range(self.population_size):

            idx = random.randint(0,self.elite_workers_num-1)
            # This is also a hyperparameter and can be changed
            a = self.epsilon * np.random.normal(0,1,size=self.weight_space)

            new_worker = elite_workers[idx] + a
            self.workers.append(new_worker) 
        
    def run(self):

        self.initial_population()
        max_reward = 0

        for i in range(self.generations):
        
            elite_workers, rewards = self.evaluate_population()

            if i == 0:
                with h5py.File(self.model_load_path, 'r') as hf:
                    elite_workers = hf[self.model_load_path][:]

            self.mutate_population(elite_workers)

            print('these are the scores', rewards, 'and this is the generation', i)
            
            self.logger.log_performance(i, rewards.mean())
            
            if rewards.mean() >= max_reward:

                max_reward = rewards.mean()
                path = '{}{:.3f}.h5'.format(self.model_save_path,max_reward)
                with h5py.File(path, 'w') as hf:
                    hf.create_dataset(path,  data=elite_workers)

                self.agent.q_estimator.save(path)
            
            elite_workers, rewards = [],[]



################################# RUN #############################################

population_size = 100
generations = 300
epsilon = 0.0001
evaluate_num = 50
elite_workers_num = 30

ga = Genetic_Algorithm(population_size, elite_workers_num, evaluate_num, generations, epsilon)    
ga.run()
    

    



