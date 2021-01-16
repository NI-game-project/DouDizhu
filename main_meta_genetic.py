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
import agents.ddqn
import agents.a2c
import pandas as pd

def set_parameters(model, weights):
    
    actor = model.actor
    critic = model.critic
    last_used = 0
    weights = tf.constant(weights[0], dtype='float32')
    
    for i in range(len(actor.layers)):

        if 'dense' in actor.layers[i].name:
            weights_shape = actor.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            actor.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if actor.layers[i].use_bias:
              weights_shape = actor.layers[i].bias.shape
              no_of_weights = tf.reduce_prod(weights_shape)
              new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
              actor.layers[i].bias = new_weights
              last_used += no_of_weights

    for i in range(len(critic.layers)):

        if 'dense' in critic.layers[i].name:
            weights_shape = critic.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            critic.layers[i].kernel = new_weights
            last_used += no_of_weights
            
            if critic.layers[i].use_bias:
              weights_shape = critic.layers[i].bias.shape
              no_of_weights = tf.reduce_prod(weights_shape)
              new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
              critic.layers[i].bias = new_weights
              last_used += no_of_weights
    
    return actor, critic
    
def initial_population(population_size, weight_space):
    
    workers = []

    fan_in = 6400*4
    
    for i in range(population_size):

        tf.keras.initializers.he_uniform()
        z = np.random.uniform(low=-np.sqrt(6/fan_in), high=np.sqrt(6/fan_in), size=weight_space)
        workers.append([z])

    return workers
    
def evaluate_population(workers, evaluate_num, config, env, random_agent, agent, rule_based_agent):

    elite_workers = []
    rewards = []
    episode_num = 20
    
    for worker in workers:
        
        agent.actor, agent.critic = set_parameters(agent, worker)
        env.set_agents([agent, rule_based_agent, rule_based_agent])

        for episode in range(episode_num):

            # Generate data from the environment
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:

                agent.feed(ts) 
       
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

    log_dir = 'experiments/genetic/a2c/a3'
    log = envs.logger.Logger(log_dir)
    
    config = {  'allow_step_back':True,
            'allow_raw_data': False, 
            'record_action': True,
            'seed': 42,
            'single_agent_mode': False,
            'active_player': True}

    population_size = 50
    
    generations = 600
    epsilon = 0.0001
    evaluate_num = 30

    #np.random.seed(42)
    weight_space = 2459197
    
    #initial_agent = agents.dqn_agent.DQNAgent()
    #model = initial_agent.q_estimator
    workers = initial_population(population_size, weight_space)

    env = envs.doudizhu.DoudizhuEnv(config)
    random_agent = agents.random_agent.RandomAgent(action_num=env.action_num)
    agent = agents.a2c.Actor_Critic(action_num=env.action_num)
    rule_based_agent = agents.doudizhu_rule_models.DouDizhuRuleAgentV1()
    max_reward = 0
    rewards = np.array([0.2,0.3])
    for i in range(generations):
        
        if i >= 1:
            elite_workers, rewards = evaluate_population(workers, evaluate_num, config, env, random_agent, agent, rule_based_agent)
        
        if i == 0:
            with h5py.File('models/genetic/a2c_a2_0.392.h5', 'r') as hf:
                elite_workers = hf['models/genetic/a2c_a2_0.392.h5'][:]
        
        workers = mutate_population(elite_workers, population_size, weight_space, epsilon)

        print('these are the scores of the first generation', rewards, 'and the generation', i)
        
        log.log_performance(i, rewards.mean())
        
        
        if rewards.mean() >= max_reward:

            max_reward = rewards.mean()
            path = 'models/genetic/a2c_a3_{:.3f}.h5'.format(max_reward)
            with h5py.File(path, 'w') as hf:
                hf.create_dataset(path,  data=elite_workers)

            #agent.q_estimator.save(path)

        

        elite_workers, rewards = [],[]
    


main()


