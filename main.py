
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
import numpy as np


import agents.a2c
import agents.ddqn_duelling
import agents.doudizhu_rule_models as doudizhu_rule_models
import agents.random_agent as random_agent
from envs.utils import set_global_seed, tournament
import envs.logger as logger
from envs.env import Env
import envs.doudizhu as doudizhu
import envs.simpledoudizhu as simpledoudizhu


#here a config dictionary has to be set, depending on what is wanted. Georg

config = {  'allow_step_back':False,
            'allow_raw_data': True, 
            'record_action': False,
            'seed': 42,
            'single_agent_mode': False,
            'active_player': False}

print('On this machine a gpu is available:', tf.test.is_gpu_available())

# Make environment
env = doudizhu.DoudizhuEnv(config)
eval_env = doudizhu.DoudizhuEnv(config)
#env = simpledoudizhu.SimpleDoudizhuEnv(config)
#eval_env = simpledoudizhu.SimpleDoudizhuEnv(config)

# Set the iterations numbers and how frequently we evaluate the performance

# TODO: These are just dummy numbers Georg
evaluate_every = 100
evaluate_num = 100
episode_num_random = 50_000
episode_num_rule = 1_000_000
save_every = 100_000
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
# TODO: Find a better way to structure the loading and storing of the models
# 

log_dir_random = 'experiments/ddqn/long_run/random'
log_dir_rule_based = 'experiments/ddqn/long_run/rule_based'

# Set a global seed
set_global_seed(42)

# Initialize a global step
global_step = tf.Variable(0, name='global_step', trainable=False)

# Set up the agents 

# uncomment the agent you want to use Georg

#agent = agents.a2c.Actor_Critic(action_num=eval_env.action_num)
#agent = agents.ddqn.DQNAgent(action_num=eval_env.action_num)
agent = agents.ddqn_duelling.DQNAgent(action_num=eval_env.action_num)
#agent = agents.a2c_ppo.Actor_Critic(action_num=eval_env.action_num)

random_agent = random_agent.RandomAgent(action_num=eval_env.action_num)
rule_based_agent = doudizhu_rule_models.DouDizhuRuleAgentV1()

# continue training with pretrained networks 
# (for actor_critic agent)

#agent.actor = keras.models.load_model('models/a2c/actor_d3.h5')
#agent.critic = keras.models.load_model('models/a2c/critic_d3.h5')

# for dqn agent

#agent.q_estimator = keras.models.load_model('models/dqn/q_estimator_v1.h5')
#agent.target_estimator = keras.models.load_model('models/dqn/target_estimator_v1.h5')


env.set_agents([agent, random_agent, random_agent])
eval_env.set_agents([agent, random_agent, random_agent])
#eval_env.set_agents([agent, rule_based_agent, rule_based_agent])

env.set_agents([agent, rule_based_agent, rule_based_agent])
eval_env.set_agents([agent, rule_based_agent, rule_based_agent])



# Init a Logger to plot the learning curve
logger_random = logger.Logger(log_dir_random)
logger_rule_based = logger.Logger(log_dir_rule_based)

for episode in range(episode_num_random):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        agent.feed(ts) 
        
    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger_random.log_performance(episode, tournament(eval_env, evaluate_num)[0],\
             #agent.history_actor, agent.history_critic, agent.optimizer._decayed_lr(tf.float32).numpy(), agent.actions, agent.predictions)
             agent.history, _, agent.optimizer._decayed_lr(tf.float32).numpy(), agent.actions, agent.predictions)
        #print(episode)
        #print(tf.reduce_sum(agent.penalty))
        #print(agent.best_actions)
        #print(agent.actions)

logger_random.close_files()
logger_random.plot('DDQN_long_run_random')

env = doudizhu.DoudizhuEnv(config)
eval_env = doudizhu.DoudizhuEnv(config)

env.set_agents([agent, rule_based_agent, rule_based_agent])
eval_env.set_agents([agent, rule_based_agent, rule_based_agent])

for episode in range(episode_num_rule):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        agent.feed(ts) 

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger_rule_based.log_performance(episode, tournament(eval_env, evaluate_num)[0],\
             #agent.history_actor, agent.history_critic, agent.optimizer._decayed_lr(tf.float32).numpy(), agent.actions, agent.predictions)
             agent.history, _, agent.optimizer._decayed_lr(tf.float32).numpy(), agent.actions, agent.predictions)
        #print(episode)
        #print(tf.reduce_sum(agent.penalty))
        #print(agent.best_actions)
        #print(agent.actions)
    
    if episode % save_every == 0:
        #agent.actor.save(('models/ddqn/actor_long_{}.h5').format(episode))
        #agent.critic.save(('models/ddqn/critic_long_{}.h5').format(episode))
        agent.q_estimator.save(('models/ddqn/long_{}.h5').format(episode))

# Close files in the logger
logger_rule_based.close_files()
# Plot the learning curve
logger_rule_based.plot('DDQN_long_run_rule_based')
# Save model
save_dir = 'models/ddqn/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# this safes the dqn models
#agent.q_estimator.save('models/dqn/q_estimator_v0_1.h5')
#agent.target_estimator.save('models/dqn/target_estimator_v0_1.h5')

#uncomment this to save the a2c model
#agent.model.save('models/a2c/ppo_a1.h5')
#agent.critic.save('models/a2c/critic_d4.h5')


    
