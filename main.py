
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


import agents.dqn_agent
import agents.actor_critic
import doudizhu_rule_models
import agents.random_agent as random_agent
from utils import set_global_seed, tournament
import logger
from env import Env
import doudizhu 


#here a config dictionary has to be set, depending on what is wanted. Georg

config = {  'allow_step_back':True,
            'allow_raw_data': True, 
            'record_action': True,
            'seed': 42,
            'single_agent_mode': False,
            'active_player': True}

# Make environment
env = doudizhu.DoudizhuEnv(config)
eval_env = doudizhu.DoudizhuEnv(config)

# Set the iterations numbers and how frequently we evaluate the performance

# TODO: These are just dummy numbers Georg
evaluate_every = 10
evaluate_num = 10
episode_num = 1000
memory_init_size = 1000

# Train the agent every X steps
train_every = 1

# The paths for saving the logs and learning curves
log_dir = './experiments/doudizhu_dqn_result/'

# Set a global seed
set_global_seed(0)

# Initialize a global step
global_step = tf.Variable(0, name='global_step', trainable=False)

# Set up the agents 
# This is going to be the place where we can add our agents Georg
# In this dummy example only dummy agents will play against each other


#agent = agents.dqn_agent.DQNAgent(action_num=eval_env.action_num) #also a random agent for now Georg
random_agent = random_agent.RandomAgent(action_num=eval_env.action_num)
actor_critic_agent = agents.actor_critic.Actor_Critic(action_num=eval_env.action_num)
rule_based_agent = doudizhu_rule_models.DouDizhuRuleAgentV1()

#continue training with pretrained networks
actor_critic_agent.actor = keras.models.load_model('DouDizhu/models/Actor.h5')
actor_critic_agent.critic = keras.models.load_model('DouDizhu/models/Critic.h5')

env.set_agents([actor_critic_agent, rule_based_agent, rule_based_agent])
eval_env.set_agents([actor_critic_agent, rule_based_agent, rule_based_agent])


# Init a Logger to plot the learning curve
logger = logger.Logger(log_dir)

for episode in range(episode_num):

    # Generate data from the environment
    trajectories, _ = env.run(is_training=True)

    # Feed transitions into agent memory, and train the agent
    for ts in trajectories[0]:
        
        #uncomment this once its not a random agent anymore Georg
        actor_critic_agent.feed(ts) 

    # Evaluate the performance. Play with random agents.
    if episode % evaluate_every == 0:
        logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])
        print(episode)

# Close files in the logger
logger.close_files()

# Plot the learning curve
logger.plot('AC')

# Save model
save_dir = 'models/doudizhu_ac_agent'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    actor_critic_agent.actor.save('Doudizhu/models/Actor_v2.h5')
    actor_critic_agent.critic.save('Doudizhu/models/Critic_v2.h5')

#TODO: create here a function which will save the model or the state of the whole training process Georg
    
