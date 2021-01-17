# DouDizhu
This is a modified version of the RLCard Toolkit which doesnt need an install and is ready to run with tensorflow 2.4, keras 2.4.3, numpy 1.19.5 
There has only been a slightly modification to the env.py file in the envs folder: the state has been enlarged by the amount of cards every player has at the current time step as well as a player id for landlord, the first peasant and the seconed peasant.

There are four agents so far, two actor critic and two dqn. (note that the a2c_ppo is NOT stable. Somehow the framework crashed without any apparent reason, mabye somebody can reread the code? I will check on it not anytime soon.) Before the experiments, the reward has to be revised because it's just adapted from the atari agents.

There also are two different implementations, which perform a genetic algorithm on the agents. The meta file contains the version which not only randomly generates the weigths of the agents, but also trains them for a predefined amout of gradient based updates before evaluating them. 

In the record.txt you can find a list of the tried experiments and the according filestructure to search for the results. 

Up next will be longer runs of all the agents as a benchmark test with similar hyperparameters and than a hyperparameter optimisation of the agents which performed best. This will also be performed for the genetic algorithm. 

Than the diversity experiments will be run: Therefore diversity measures for the weightspave have to be selected. One would be the cosine similarity as well as calculation of entropy. t-SNE comes into mind as well as PCA. Suggestions are more than welcome. Aditionally the outputs of the network will be compared: agreement on taken action as well as cross entropy of probabilities.

Lastly the performence of ensembles will be analyzed. And especially, if the ensemble work better if its members are similar or diverse. All these things will be done for a simple image classifier and the RL agents of DouDizhu (and maybe also on some atari games, but its not sure if this will have any new findings).

(Try the muzero approach if there still is time)