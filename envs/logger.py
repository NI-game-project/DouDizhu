import os
import csv

class Logger(object):
    ''' Logger saves the running results and helps make plots from the results
    '''

    def __init__(self, log_dir):
        ''' Initialize the labels, legend and paths of the plot and log file.

        Args:
            log_path (str): The path the log files
        '''
        self.log_dir = log_dir
        self.txt_path = os.path.join(log_dir, 'log.txt')
        self.csv_path = os.path.join(log_dir, 'performance.csv')
        self.fig_path = os.path.join(log_dir, 'fig.png')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.txt_file = open(self.txt_path, 'w')
        self.csv_file = open(self.csv_path, 'w')
        fieldnames = ['generation', 'reward', 'actor_loss', 'critic_loss', 'learning_rate', 'actions', 'predictions']
        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

    def log(self, text):
        ''' Write the text to log file then print it.
        Args:
            text(string): text to log
        '''
        self.txt_file.write(text+'\n')
        self.txt_file.flush()
        print(text)

    def log_performance(self, timestep, reward, actor_loss, critic_loss, learning_rate, actions, predictions):
        ''' Log a point in the curve
        Args:
            timestep (int): the timestep of the current point
            reward (float): the reward of the current point
        '''
        self.writer.writerow({'generation': timestep,
                             'reward': reward,
                             'actor_loss': actor_loss,
                             'critic_loss': critic_loss,
                             'learning_rate': learning_rate,
                             'actions': actions,
                             'predictions': predictions})
        print('')
        self.log('----------------------------------------')
        self.log('  generation   |  ' + str(timestep))
        self.log('  reward       |  ' + str(reward))
        self.log('  actor loss   |  ' + str(actor_loss))
        self.log('  criic loss   |  ' + str(critic_loss))
        self.log('  learningrate |  ' + str(learning_rate))
        self.log('  actions      |  ' + str(actions))
        self.log('  predictions  |  ' + str(predictions))
        self.log('----------------------------------------')

    def plot(self, algorithm):
        plot(self.csv_path, self.fig_path, algorithm)

    def close_files(self):
        ''' Close the created file objects
        '''
        if self.txt_path is not None:
            self.txt_file.close()
        if self.csv_path is not None:
            self.csv_file.close()

def plot(csv_path, save_path, algorithm):
    ''' Read data from csv file and plot the results
    '''
    import matplotlib.pyplot as plt
    with open(csv_path) as csvfile:
        print(csv_path)
        reader = csv.DictReader(csvfile)
        xs = []
        ys = []
        for row in reader:
            xs.append(int(row['generation']))
            ys.append(float(row['reward']))
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=algorithm)
        ax.set(xlabel='generation', ylabel='reward')
        ax.legend()
        ax.grid()

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        fig.savefig(save_path)
