import gym 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow_core.python.keras import layers,Sequential,regularizers,datasets,metrics,losses
from tensorflow_core.python.keras.api._v2.keras import optimizers
class Actor(keras.Model):
    def __init__(self):
        super(Actor,self).__init__()
        self.fc1=layers.Dense(100,kernel_initializar='he_normal')
        self.fc2=layers.Dense(2,kernel_initializar='he_normal')
    def call(self,inputs):
        x=tf.nn.relu(self,fc1(inputs))
        x=self.fc2(x)
        x=tf.nnsoftmax(x,axis=1)
        return x

class Critic(keras.Model):
    def __init__(self):
        super(Critic,self).__init__()
        self.fc1=layers.Dense(100,kernel_initializar='he_normal')
        self.fc2=layers.Dense(2,kernel_initializar='he_normal')
    def call(self,inputs):
        x=tf.nn.relu(self,fc1(inputs))
        x=self.fc2(x)
        x=tf.nnsoftmax(x,axis=1)
        return x

class PPO(object):
    def __init__(self):
        super(PPO,self).__init__()
        self.actor=Actor()
        self.critic=Critic()
        self.buffer=[]
        self.actor_optimizer=optimizers.Adam(1e-3)
        self.critic_optimizer=optimizers.Adam(1e-3)
    def select_action(self,s):
        s=tf.constant(s,dtype=tf.float32)
        s=tf.expend_dims(s,axis=0)
        prob=self.actor(s)
        a=tf.random.categorical(tf.math.log(porb),1)[0]
        a=int(a)
        return a,float(prob[0][a])
    def optimize(self,gamma,batch_size,epsilon):
        state=tf.constant([t.state for t in self.buffer],dtype=tf.float32)
        action=tf.constant([t.action for t in self.buffer],dtype=tf.float32)
        action=tf.reshape(action,[-1,1])
        reward=[t.reward for t in self.buffer]
        old_action_log_prob=tf.constant([t.a_log_prob for t in self.buffer],dtype=tf.float32)
        old_action_log_prob=tf.reshape(old_action_log_prob,[-1,1])
        R=0
        Rs=[]
        for r in reward[::-1]:
            R=r+gamma*R
            Rs.insert(0,R)
        Rs=tf.constant(Rs,dtype=tf.float32)
        for _ in range(round(10*len(self.buffer)/batch_size)):
            index=np.random.choice(np.arange(len(self.buffer)),batch_size,replace=False)
            with tf.GradientTape()as tape1,tf.GradientTape() as tape2:
                v_target=tf.expend_dims(tf.gather(Rs,index,axis=0),axis=1)
                v=self.critic(tf.gather(state,index,axis=0))
                delta=v_target-v
                advantage=tf.stop_gradient(delta)
                a=tf.gather(action,index,axis=0)
                pi=self.actor(tf.gather(state,index,axis=0))
                indices=tf.expand_dims(tf.range(a.shape[0]),axis=1)
                indices=tf.concat([indices,a],axis=1)
                pi_a=tf.gather_nd(pi,indices)
                pi_a=tf.expand_dims(pi_a,axis=1)
                ratio=(pi_a/tf.gather(old_action_log_prob,index,axis=0))
                surr1=ratio*advantage
                surr2=tf.clip_by_value(ratio,1-epsilon,1+epsilon)*advantage
                policy_loss=-tf.reduce_mean(tf.minimum(surr1,surr2))
                value_loss=losses.MSE(v_target,v)
                grads=tape1.gradient(policy_loss,self.actor.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads,self.actor.trainable_variables))
                grads=tape2.gradient(policy_loss,self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads,self.critic.trainable_variables))
        self.buffer=[]

calss St(object):


def main():
    agent=PPO()
    returns=[]
    total=0
    for i_epoch in range(500):
        state=env.reset()
        for t in range(500):
            action,action_prob=agent.select_action(state)
            next_state,reward,done,_=env.step(action)
            trans=


