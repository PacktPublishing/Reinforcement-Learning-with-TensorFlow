# coding: utf-8

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

def preprocessing_image(obs): #where I is the single frame of the game as the input
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    #the values below have been precomputed through trail and error by OpenAI team members
    obs = obs[35:195] #cropping the image frame to an extent where it contains on the paddles and ball and area between them
    obs = obs[::2,::2,0] #downsample by the factor of 2 and take only the R of the RGB channel.Therefore, now 2D frame
    obs[obs==144] = 0 #erase background type 1
    obs[obs==109] = 0 #erase background type 2
    obs[obs!=0] = 1 #everything else(other than paddles and ball) set to 1
    return obs.astype('float').ravel() #flattening to 1D


game_env = 'Pong-v0'
num_workers = multiprocessing.cpu_count()
max_global_episodes = 100000
global_network_scope = 'globalnet'
global_iteration_update = 20
gamma = 0.9
beta = 0.0001
lr_actor = 0.0001    # learning rate for actor
lr_critic = 0.0001    # learning rate for critic
global_running_rate = []
global_episode = 0

env = gym.make(game_env)

num_actions = env.action_space.n


tf.reset_default_graph()


class ActorCriticNetwork(object):
    def __init__(self, scope, globalAC=None):
        if scope == global_network_scope:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None,6400], 'state')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None,6400], 'state')
                self.a_his = tf.placeholder(tf.int32, [None,], 'action')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'target_vector')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='temporal_difference_error')
                with tf.name_scope('critic_loss'):
                	self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('actor_loss'):
                	log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, num_actions, dtype=tf.float32), axis=1, keep_dims=True)
                	exp_v = log_prob * td
                	entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                			     axis=1, keep_dims=True)  #exploration
                	self.exp_v = beta * entropy + exp_v
                	self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                	self.a_grads = tf.gradients(self.a_loss, self.a_params)
                	self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = actor_train.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = critic_train.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor_network'):
            l_a = tf.layers.dense(self.s, 300, tf.nn.relu6, kernel_initializer=w_init, name='actor_layer')
            a_prob = tf.layers.dense(l_a, num_actions, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic_network'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='critic_layer')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run local
        session.run([self.update_a_op, self.update_c_op], feed_dict)  # local gradient applied to global net

    def pull_global(self):  # run local
        session.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run local
        s = np.reshape(s,[-1])
        prob_weights = session.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action



class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(game_env).unwrapped
        self.name = name
        self.AC = ActorCriticNetwork(name, globalAC)

    def work(self):
        global global_running_rate, global_episode
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not coordinator.should_stop() and global_episode < max_global_episodes:
            obs = self.env.reset()
            s = preprocessing_image(obs)
            ep_r = 0
            while True:
                if self.name == 'W_0':
                    self.env.render()
                a = self.AC.choose_action(s)

                #print(a.shape)

                obs_, r, done, info = self.env.step(a)
                s_ = preprocessing_image(obs_)
                if done and r<=0: 
                    r = -20
                ep_r += r
                buffer_s.append(np.reshape(s,[-1]))
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % global_iteration_update == 0 or done:   # update global and assign to local net
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        s_ = np.reshape(s_,[-1])
                        v_s_ = session.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + gamma * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                    }
                    self.AC.update_global(feed_dict)

                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    if len(global_running_rate) == 0:  # record running episode reward
                        global_running_rate.append(ep_r)
                    else:
                        global_running_rate.append(0.99 * global_running_rate[-1] + 0.01 * ep_r)
                    print(
                        self.name,
                        "Ep:", global_episode,
                        "| Ep_r: %i" % global_running_rate[-1],
                          )
                    global_episode += 1
                    break


if __name__ == "__main__":
    session = tf.Session()

    with tf.device("/cpu:0"):
        actor_train = tf.train.RMSPropOptimizer(lr_actor, name='RMSPropOptimiserActor')
        critic_train = tf.train.RMSPropOptimizer(lr_critic, name='RMSPropOptimiserCritic')
        acn_global = ActorCriticNetwork(global_network_scope)  # we only need its params
        workers = []
        # Create worker
        for i in range(num_workers):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, acn_global))

    coordinator = tf.train.Coordinator()
    session.run(tf.global_variables_initializer())

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coordinator.join(worker_threads)

    plt.plot(np.arange(len(global_running_rate)), global_running_rate)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.show()

