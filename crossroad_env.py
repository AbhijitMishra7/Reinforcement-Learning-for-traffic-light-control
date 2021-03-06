# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 23:00:57 2022

@author: abhij
"""

import tensorflow as tf
from gym import Env
from gym.spaces import Discrete, Box, Tuple
import numpy as np
import random
import collections
import matplotlib.pyplot as plt

class Crossroad_Env(Env):
    
    def __init__(self):
        #action space vector of size 4 indicating red(0) or green(1) light 
        self.action_space=Tuple((Discrete(2),Discrete(2),Discrete(2),Discrete(2)))
        #observation space vector of size 4 indicating the level of traffic on 
        # a particular lane with range[0,1] 0=>No traffic 1=>Max traffic
        self.observation_space=Box(low=0,high=1,shape=(4,))
        self.state=np.random.uniform(0.1,0.3,size=(4,)) 
        #timer is used to keep the task episodic. Each time step is the 
        #time interval between switching of the colors of traffic light 
        self.timer=20
    def step(self,action):
        done=False
        info={}
        self.timer-=1
        if(self.timer==0):
            done=True
        reward=0
        #penalty for the current traffic state
        current_state=tf.constant(self.state)
        
        #reward=-(np.sum(self.state)/2)
        #reward for signalling green light
        #reward=np.sum(action)
        #penalty for causing high traffic on the crossroad
        if((np.sum(action)>1)&(np.sum(action*self.state)>0.7)):
            reward-=5
        #change of state of the lanes during the time interval according to color
        # of light on the respective lane
        for i in range(4):
            if((action[i]==1)&(self.state[i]>0.8)):
                self.state[i]-=0.8
            elif((action[i]==1)&(self.state[i]<=0.8)):
                self.state[i]=0
            else:
                pass
        #Penalty for future traffic state
        future_state=tf.constant(self.state)
        reward+=np.sum(tf.subtract(current_state, future_state).numpy())
        #reward-=(np.sum(self.state)/2)
        #New state of each lane
        for i in range(4):
            self.state[i]+=random.uniform(0.0, 0.1)
            if(self.state[i]>1):
                self.state[i]=1
        
        return self.state,reward,done,info
    def reset(self):
        self.state=np.random.uniform(0.1,0.3,size=(4,)) 
        self.timer=20
        return self.state
 
#Agent Implementation
from keras.models import Model
from keras.layers import Dense, Input, Layer, Concatenate, LayerNormalization
from keras.optimizers import Adam

class Normalization_Layer(Layer):

    def __init__(self):
        super(Normalization_Layer, self).__init__()
        self.norm=LayerNormalization()
    def call(self, inputs):
        normalized=self.norm(inputs)
        out=normalized/2+0.5;
        return out

class Value_fun(Layer):
    def __init__(self):
        super(Value_fun,self).__init__()
        self.dense1=Dense(16,'relu')
        self.dense2=Dense(32,'relu')
        self.dense3=Dense(16,'relu')
        self.dense4=Dense(4,'relu')
        self.normalize=Normalization_Layer()
        self.concat=Concatenate()
        self.out=Dense(16)
    def call(self,inp):
        x=self.dense1(inp)
        y1=self.dense2(x)
        y2=self.dense3(y1)
        y2=self.dense4(y2)
        y2=self.normalize(y2)
        y2=self.concat([inp,y2])
        output=self.out(y2)
        return output

#Epsilon greedy policy    
def policy_fun(q,state,epsilon=0.1):
    action_prob = np.ones(16, dtype=float) * epsilon / 16
    best_action=np.argmax(q(tf.expand_dims(state,0)))
    action_prob[best_action]+=(1-epsilon)
    return action_prob

def action_selection(action_prob,q,state,epsilon):
    if(random.uniform(0,1)<epsilon):
        return random.randint(0, 15)
    else:
        return np.argmax(q(tf.expand_dims(state,0)))


class ExperienceReplay:
  def __init__(self, capacity):
      self.buffer = collections.deque(maxlen=capacity)
  def __len__(self):
      return len(self.buffer)
  def append(self, experience):
      self.buffer.append(experience)
  
  def sample(self, batch_size):
      states=np.empty([batch_size, 4])
      actions=np.empty([batch_size, 1],dtype=int)
      rewards=np.empty([batch_size, 1])
      dones=np.empty([batch_size, 1],dtype=bool)
      next_states=np.empty([batch_size, 4])
      batch=random.sample(self.buffer,batch_size)
      for j in range(batch_size):
          state,action,reward,done,next_state=batch[j]
          states[j]=state
          actions[j]=action
          rewards[j]=reward
          dones[j]=done
          next_states[j]=next_state
      return states, actions,rewards,dones,next_states

def loss_fun(q,gamma, state, action, reward, done, next_state):
    if(done):
        loss=(reward-q(tf.expand_dims(state,0))[0,action])**2
    else:
        loss=(reward+gamma*(np.max(q(tf.expand_dims(next_state,0))))-q(tf.expand_dims(state,0))[0,action])**2
    return loss

def scalar_to_action_space(v):
    return np.array(list(map(int, (bin(v).replace("0b", "").zfill(4)))))

rewards_list=[]
loss_list=[]

def run_model(env,q,q_,epsilon,alpha,gamma, Experience,episodes):
    replay_buffer=ExperienceReplay(1000)
    for _ in range(episodes):
        if(_%10==0):
            print('episode {}'.format(_))
        done=False
        losses_printed=False
        env.reset()
        while(not done):                
            current_state=tf.constant(env.state)
            action_prob=policy_fun(q, env.state)
            action=action_selection(action_prob, q, env.state, epsilon)
            next_state,reward,done,info=env.step(scalar_to_action_space(action))
            replay_buffer.append(Experience(current_state.numpy(),action,reward,done,next_state))
            value_fun_sync=100
            no_of_updates=0
            if(len(replay_buffer)==1000):
                no_of_updates+=1
                if(no_of_updates%value_fun_sync==0):
                    q_=q
                states,actions,rewards,dones,next_states=replay_buffer.sample(64)
                optimizer=Adam()
                with tf.GradientTape() as tape:
                    loss=[]
                    for i in range(64):
                        loss.append(loss_fun(q_, gamma, states[i], actions[i][0], rewards[i][0], dones[i][0], next_states[i]))                        
                if(not losses_printed):
                    losses_printed=True
                    print('Sum of rewards: {}    Loss: {}'.format(np.sum(rewards),np.sum(loss)))
                    rewards_list.append(np.sum(rewards))
                    loss_list.append(np.sum(loss))
                grad=tape.gradient(loss,q.trainable_variables)
                optimizer.apply_gradients(zip(grad,q.trainable_variables))
            else:
                pass


env=Crossroad_Env()
inp=Input((4,))
value_fun=Value_fun()(inp)
q=Model(inputs=inp,outputs=value_fun)
q(tf.expand_dims(env.state,0))
q_=q
epsilon=0.1
alpha=1e-3
gamma=0.99

Experience = collections.namedtuple('Experience', 
           field_names=['state', 'action', 'reward','done', 'new_state'])


run_model(env, q, q_, epsilon, alpha, gamma, Experience, 500)

'''
env.reset()
print(env.state)
actionsProb=q(tf.expand_dims(env.state,0))
print(actionsProb)
actionTaken=scalar_to_action_space(np.argmax(actionsProb))
print(actionTaken)
next_state,reward,done,info=env.step(actionTaken)
print(env.state)
q.get_weights()
'''

plt.xlabel('Number of episodes')
plt.ylabel('Sum of rewards per episode')
plt.plot(rewards_list)


import seaborn as sns
sns.set()


numbers = rewards_list
window_size = 50

i = 0
moving_averages = []
while i < len(numbers) - window_size + 1:
    this_window = numbers[i : i + window_size]
    window_average = sum(this_window) / window_size
    moving_averages.append(window_average)
    i += 1

plt.xlabel('Number of episodes')
plt.ylabel('Sum of rewards per episode')
plt.plot(rewards_list)
plt.plot(moving_averages,color='red')

