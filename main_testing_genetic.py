
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


import tensorflow as tf
import keras
import os

from keras.optimizers import Adam
import agents.ddqn
import agents.a2c
import agents.doudizhu_rule_models
import agents.random_agent as random_agent
from envs.utils import set_global_seed, tournament
import envs.logger
from envs.env import Env
import envs.doudizhu 
import h5py


#here a config dictionary has to be set, depending on what is wanted. Georg

config = {  'allow_step_back':True,
            'allow_raw_data': True, 
            'record_action': True,
            'seed': 0,
            'single_agent_mode': False,
            'active_player': True}

# Make environment
env = envs.doudizhu.DoudizhuEnv(config)
eval_env = envs.doudizhu.DoudizhuEnv(config)

# Set the iterations numbers and how frequently we evaluate the performance

# TODO: These are just dummy numbers Georg
evaluate_every = 50
evaluate_num = 50
episode_num = 1000000
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
# TODO: Find a better way to structure the loading and storing of the models
# 

log_dir = 'experiments/a2c/z3'

# Set a global seed
set_global_seed(42)

# Initialize a global step
global_step = tf.Variable(0, name='global_step', trainable=False)

# Set up the agents 

# uncomment the agent you want to use Georg

agent = agents.ddqn_agent.DQNAgent(action_num=eval_env.action_num) 
#agent = agents.a2c.Actor_Critic(action_num=eval_env.action_num)

random_agent = random_agent.RandomAgent(action_num=eval_env.action_num)
rule_based_agent = agents.doudizhu_rule_models.DouDizhuRuleAgentV1()

# continue training with pretrained networks 
# (for actor_critic agent)

#agent.actor = keras.models.load_model('models/a2c/actor_s5.h5')
#agent.critic = keras.models.load_model('models/a2c/critic_s5.h5')

# for dqn agent

#agent.q_estimator = keras.models.load_model('models/dqn/q_estimator_g0.h5')
#agent.target_estimator = keras.models.load_model('models/dqn/target_estimator_g0.h5')

# uncomment this for the loading of the genetic weights

def set_agent():
    with h5py.File('models/genetic/d3_0.491.h5', 'r') as hfxx:
        dataxx = hfxx['models/genetic/d3_0.491.h5'][:]
    
    weights = dataxx[9]

    model = agent.q_estimator
    print(model.summary())
    last_used = 0
    weights = tf.constant(weights[0], dtype='float32')

    for i in range(len(model.layers)):

        if i == 20:
            
            batch_weights_beta = tf.reshape(weights[:450], (450))
            batch_weights_gamma = tf.reshape(weights[450:900], (450))
            batch_weights_mean = tf.reshape(weights[900:1350], (450))
            batch_weights_var = tf.reshape(weights[1350:1800], (450))
            model.layers[i].set_weights([batch_weights_beta, batch_weights_gamma, batch_weights_mean, batch_weights_var])
            last_used += 1800

        if 'dense' in model.layers[i].name:
            weights_shape = model.layers[i].kernel.shape
            no_of_weights = tf.reduce_prod(weights_shape)
            new_weights = tf.reshape(weights[last_used:last_used+no_of_weights], weights_shape) 
            #model.layers[i].set_weights(new_weights)
            
            last_used += no_of_weights
        
            weights_shape_bias = model.layers[i].bias.shape
            no_of_weights_bias = tf.reduce_prod(weights_shape_bias)
            new_weights_bias = tf.reshape(weights[last_used:last_used+no_of_weights_bias], weights_shape_bias) 
            model.layers[i].set_weights([new_weights, new_weights_bias])
            last_used += no_of_weights_bias
            print(last_used)
            print(len(weights))
        

    return model

#agent.q_estimator = set_agent()
#agent.target_estimator = set_agent()



#agent.q_estimator.build()
#agent.target_estimator.compile(loss='mse', optimizer=Adam(lr=learning_rate))


env.set_agents([agent, agent, agent])
eval_env.set_agents([agent, agent, agent])


#env.set_agents([agent, random_agent, random_agent])
#eval_env.set_agents([agent, random_agent, random_agent])

#env.set_agents([agent, rule_based_agent, rule_based_agent])
#eval_env.set_agents([agent, rule_based_agent, rule_based_agent])


# Init a Logger to plot the learning curve

# TODO: restructure this 
logger = envs.logger.Logger(log_dir)

#logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)
    
    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:

        agent.feed(ts) 

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
        print(episode)
        #print(tf.reduce_sum(agent.penalty))
        #print(agent.best_actions)
        #print(agent.actions)

episode_num = 100000

env.set_agents([agent, rule_based_agent, rule_based_agent])
eval_env.set_agents([agent, rule_based_agent, rule_based_agent])
for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:

        agent.feed(ts) 

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
        print(episode)
        #print(tf.reduce_sum(agent.penalty))
        #print(agent.best_actions)

# Close files in the logger
logger.close_files()

# Plot the learning curve
# TODO: restructure this
logger.plot('test')

# Save model
save_dir = 'models/a2c/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



# this safes the dqn models
#agent.q_estimator.save('models/dqn/q_estimator_test.h5')
#agent.target_estimator.save('models/dqn/target_estimator_test.h5')

#uncomment this to save the a2c model
agent.model.save('models/a2c/actor_z3.h5')



    
