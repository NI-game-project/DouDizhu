import os
import random
import gym
import pylab
import numpy as np
import tensorflow as tf 
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense, Lambda, Add, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import cv2
import networks


class Muzero_Agent:

    def __init__(self, env_name, setup):

        self.setup = setup
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.output_shape = self.env.action_space.n 

        self.episodes = 1000
        self.epochs = 10
        self.batchsize = 32
        self.lr = 0.00005

        self.rows = 80
        self.cols = 80
        self.channels = 1

        self.states, self.actions, self.rewards, self.predictions = [], [], [], []
        self.scores, self.average = [], []

        self.Save_Path = 'models/pong_hypernetwork'
        self.input_shape = (self.channels, self.rows, self.cols)
        self.image_memory = np.zeros(self.input_shape)

        self.embedding_size = 100

        h_input = self.input_shape
        h = Dense(64, activation='relu')(h_input)
        h = Dense(self.embedding_size, activation='relu')(h)
        self.h = h

        g_input = self.embedding_size
        g = Dense(64, activation='relu')(g_input)
        g = Dense(self.output_shape, activation='relu')(g)
        self.g = g

        p_input = self.embedding_size
        p = Dense(64, activation='relu')(p_input)
        p = Dense(self.output_shape, activation='relu')(p)
        self.p = p

        v_input = self.embedding_size
        v = Dense(64, activation='relu')(v_input)
        v = Dense(self.output_shape, activation='relu')(v)
        self.v = v

    
    def feed_memory(self, state, action, reward, next_state, done):
  
        self.memory.save(state, action, reward, next_state, done)


    def normal_run(self):

        optimizer = RMSprop(lr=self.lr)
        
        for i in range(self.episodes):
            
            frame = self.env.reset()
            state = self.GetImage(frame)
            
            done = False
            score = 0

            while not done:
                self.env.render()

                prediction = self.Actor(state, training=True)[0]
                self.predictions.append(prediction)
                
                action = np.random.choice(self.output_shape, p=prediction.numpy())
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.GetImage(next_state)

                self.states.append(state)
                action_onehot = np.zeros([self.output_shape])
                action_onehot[action] = 1
                self.actions.append(action_onehot)
                self.rewards.append(reward)
                state = next_state
                score += reward
                
                if done:

                    # reshape memory to appropriate shape for training
                    self.states = np.vstack(self.states)
                    self.actions = np.vstack(self.actions)
                    self.predictions = np.vstack(self.predictions)
                    #self.actions = tf.convert_to_tensor(self.actions)

                    # Compute discounted rewards
                    discounted_r = np.vstack(self.discount_rewards(self.rewards))

                    # Get Critic network predictions
                    values = self.Critic(self.states, training=True)

                    # Compute advantages
                    self.advantages = discounted_r - values

                    y_true = np.hstack([self.advantages, self.predictions, self.actions])


                    for e in range(self.epochs):

                        with tf.GradientTape() as tape:
                            np.random.seed(e)
                            np.random.shuffle(y_true)
                            np.random.seed(e)
                            np.random.shuffle(self.states)
            
                            advantages, predictions, actions = y_true[:, :1], y_true[:, 1:1+self.output_shape], y_true[:, 1+self.output_shape:]

                            y_pred_actor = self.Actor(self.states, training=True)
                            loss_clipping = 0.2
                            entropy_loss = 5e-3

                            prob = y_pred_actor * actions
                            old_prob = actions * predictions
                            r = prob/(old_prob + 1e-10)
                            p1 = r * advantages
                            p2 = K.clip(r, min_value=1-loss_clipping, max_value=1+loss_clipping) * advantages
                            loss = - K.mean(K.minimum(p1,p2) + entropy_loss * -(prob*K.log(prob + 1e-10)))

                            
                            grads = tape.gradient(loss, self.Actor.trainable_weights)
                            optimizer.apply_gradients(zip(grads, self.Actor.trainable_weights))

                            #self.Actor.fit(self.states, y_true, epochs=self.EPOCHS, verbose=0, shuffle=True, batch_size=len(self.rewards))
                            self.Critic.fit(self.states, discounted_r, epochs=1, verbose=0, shuffle=True, batch_size=len(self.rewards))



                    # reset training memory
                    print("this is the reward", score, 'episode', i)
                    self.states, self.actions, self.rewards, self.predictions, self.advantages = [], [], [], [], []

                    #print("episode: {}/{}, score: {}, average: {:.2f} {}".format(e, self.EPISODES, score, average, SAVING))
                        
                        
        self.env.close()



    def step(self, action):

        next_state, reward, done, _ = self.env.step(action)
        next_state = self.GetImage(next_state)

        return next_state, reward, done

    def discount_rewards(self, reward):
        # Compute the gamma-discounted rewards over an episode
        gamma = 0.997    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
                running_add = 0
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add

        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    
    def GetImage(self, frame):
        # croping frame to 80x80 size
        frame_cropped = frame[35:195:2, ::2,:]
        if frame_cropped.shape[0] != self.cols or frame_cropped.shape[1] != self.rows:
            # OpenCV resize function 
            frame_cropped = cv2.resize(frame, (self.cols, self.rows), interpolation=cv2.INTER_CUBIC)
        
        # converting to RGB (numpy way)
        frame_rgb = 0.299*frame_cropped[:,:,0] + 0.587*frame_cropped[:,:,1] + 0.114*frame_cropped[:,:,2]

        # convert everything to black and white (agent will train faster)
        frame_rgb[frame_rgb < 100] = 0
        frame_rgb[frame_rgb >= 100] = 255    

        # dividing by 255 we expresses value to 0-1 representation
        new_frame = np.array(frame_rgb).astype(np.float32) / 255.0
        new_frame = new_frame[np.newaxis, np.newaxis,:,:]

        return new_frame 


class Memory(object):


    def __init__(self, memory_size, batch_size):
 
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, done):

        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self):

        samples = random.sample(self.memory, self.batch_size)

        return map(np.array, zip(*samples))



# representation function


# dynamic function


# prediction function (value and policy function)