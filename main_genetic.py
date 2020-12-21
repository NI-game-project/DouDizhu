import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense
from keras.optimizers import Adam, RMSprop

#import agents.genetic_algorithm
import agents.random_agent as random_agent
import simpledoudizhu
import logger
from utils import set_global_seed, tournament
import agents.dqn_agent


def set_parameters(model, weights):
    
    last_used = 0
    for i in range(len(model.layers)):

        if 'dense' in model.layers[i].name: 
            weights_shape = model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            model.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if model.layers[i].use_bias:
              weights_shape = model.layers[i].bias.shape
              no_of_weights = tf.reduce_prod(weights_shape)
              new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
              model.layers[i].bias = new_weights
              last_used += no_of_weights
    
    return model
    
def initial_population(population_size, weight_space):
    
    workers = []
    kernel1 = 13107712 - 512
    #kernel1 = 3277312 - 512
    bias1 = kernel1 + 512
    kernel2 = bias1 + 3072
    bias2 = kernel2 + 6

    fan_in = 6400*4
    
    for i in range(population_size):

        tf.keras.initializers.he_uniform()
        z = np.random.uniform(low=-np.sqrt(6/fan_in), high=np.sqrt(6/fan_in), size=weight_space)
        z[kernel1:bias1] = np.zeros(bias1-kernel1)
        z[kernel2:bias2] = np.zeros(bias2-kernel2)
        workers.append([z])

    return workers
    
def evaluate_population(workers, evaluate_num, config):

    elite_workers = []
    rewards = []
    env = simpledoudizhu.SimpleDoudizhuEnv(config)
    random_agent = random_agent.RandomAgent(action_num=env.action_num)

    for worker in workers:
        
        agent = agents.dqn_agent.DQNAgent(action_num=env.action_num)
        env.set_agents([agent, random_agent, random_agent])

        payoff = tournament(env, evaluate_num)[0]

        rewards.append(payoff)
    
    rewards = np.array(rewards)

    elite_idx = np.argsort(rewards)[40:]

    for idx in elite_idx:
        elite_workers.append(workers[idx])

    return elite_workers, rewards

def mutate_population(elite_workers,population_size, weight_space, epsilon):

    workers = []

    for i in range(population_size):

        idx = random.randint(0,9)
        a = epsilon * np.random.normal(0,1,size=weight_space)

        new_worker = elite_workers[idx] + a
        workers.append(new_worker) 
    
    return workers


def main():

    log_dir = './experiments/genetic/a1'
    log = logger.Logger(log_dir)
    
    config = {  'allow_step_back':True,
            'allow_raw_data': True, 
            'record_action': True,
            'seed': 42,
            'single_agent_mode': False,
            'active_player': True}

    population_size = 50
    
    generations = 300
    epsilon = 0.002
    evaluate_num = 30
    


    #np.random.seed(42)
    weight_space = 2459197
    workers = initial_population(population_size, weight_space)
    
    for i in range(generations):

        elite_workers, rewards = evaluate_population(workers, evaluate_num, config)
        workers = []
        workers = mutate_population(elite_workers, population_size, weight_space, epsilon)

        print('these are the scores of the first generation', rewards, 'and the generation', i)
        elite_workers, rewards = [],[]

main()


