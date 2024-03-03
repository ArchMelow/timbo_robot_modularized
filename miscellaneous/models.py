import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, concatenate
from keras.optimizers import Adam
import numpy as np
import random
import matplotlib.pyplot as plt
from serial_gui import SerialComm
import json
import time
from tqdm import tqdm
from copy import deepcopy

# Replay buffer
import numpy as np
from collections import deque
import random

FAIL = -1
PROG = 0
SUCCESS = 1



class ReplayBuffer(object):
    """
    Reply Buffer
    """
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = deque()
        self.count = 0

    def add_buffer(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)

        if self.count < self.buffer_size:
            self.buffer.append(transition)
            self.count += 1
        else: 
            self.buffer.popleft()
            self.buffer.append(transition)

    def sample_batch(self, batch_size):
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        
        states = np.asarray([i[0] for i in batch])
        actions = np.asarray([i[1] for i in batch])
        rewards = np.asarray([i[2] for i in batch])
        next_states = np.asarray([i[3] for i in batch])
        dones = np.asarray([i[4] for i in batch])
        return states, actions, rewards, next_states, dones


    
    def buffer_count(self):
        return self.count

    def clear_buffer(self):
        self.buffer = deque()
        self.count = 0

# environment
class Env(object):
    def __init__(self, comm_obj):
        # SHOULD MODIFY (DISTANCE SENSOR - 21 dim later)
        self.state_space_dim = 21 # ([ax, ay, az, mx, my, mz, gx, gy, gz, adc1, ..])
        self.action_space_dim = 2 # ([adc1, adc2])
        self.action_bound = 400 # ([0, 1296])
        
        self.prev_pos = None # init
        
        self.comm_obj = comm_obj
        self.end_loop_flag = False
        self.ready_to_write = False
        
        # head-tail reversed?
        self.head_tail_order_rev = False
    
    def normalize_state_vector(self, vector):
        # normalize a vector to range [-1, 1]
        # minmax normalization
        vector = np.array(vector)
        maxi = np.max(vector); mini = np.min(vector)
        if maxi != mini:
            return (2.*(vector - mini)/(maxi - mini)) -1.
        elif mini == maxi and mini == 0:
            return vector    
        else: # avoid division by zero
            return vector / mini
            
    
    
    def block_until_recv(self):
        
        if isinstance(self.comm_obj, SerialComm):
            s = self.comm_obj.serialObj
            while not s.isOpen() or not s.in_waiting:
                # block until something arrives
                self.comm_obj.root.update()
                self.comm_obj.dataCanvas.config(scrollregion=self.comm_obj.dataCanvas.bbox("all"))
            
            recentPacket = s.read(1) # read one byte
            recentPacketString = None
            data = None
            
            # read until the packet becomes a valid JSON string
            while True:
                recentPacketString =recentPacket.decode('utf-8').rstrip('\n')
                try:
                    data = json.loads(recentPacketString)
                except:
                    recentPacket += s.read(1)
                else:
                    break
                #raise ValueError('send the packet in JSON format.')
            
            return data
    
    # send a msg to the ESP32 and get back a response
    def exchange_serial_msg(self, msg):
        if isinstance(self.comm_obj, SerialComm):
            while True:
                self.comm_obj.root.update()
                #print(self.comm_obj.serialObj.port)
                #print(self.comm_obj.serialObj.isOpen())
                if self.comm_obj.serialObj.isOpen():
                    
                    a = self.comm_obj.serialObj.write(msg)
                    break
                self.comm_obj.dataCanvas.config(scrollregion=self.comm_obj.dataCanvas.bbox("all"))
            
            
            #d = self.comm_obj.checkSerialPort()
            d = self.block_until_recv()
            
            #self.comm_obj.serialObj.reset_input_buffer()
            #self.comm_obj.serialObj.reset_output_buffer()
            #self.comm_obj.serialObj.flush()        
            return d # return response
    
    # order data tuple to [head, tail] order    
    # tail needs to be at 3154 (initial state) 
    # head needs to be at 1694 (initial state)   
    def order_data_pair(self, data_pair):
        data0, data1 = data_pair
        if len(data0) == 10:
            return [data1, data0]
        if len(data1) == 10:
            return [data0, data1]
            
    
    def return_reward(self, data_lists = None):
        data0, data1 = self.order_data_pair(data_lists)

        print(data0, data1)
        # data format (index)
        # 0,1,2 : ax,ay,az
        # 3,4,5 : mx,my,mz
        # 6,7,8 : gx,gy,gz
        # 9 : adc
        # 10 : tof (dist) - only for the head slave
        
        # if the robot lost its balance, penalize. (-100)
        if (abs(data0[0]) >= 0.55 or abs(data0[1]) >= 0.55) \
            or (abs(0.68- data1[0]) >= 0.55 or abs(data1[1]) >= 0.55):
            return -1000
        
        # penalize for backward movement, reward for forward movement.
        
        # if it's the first time to collect distance info
        cur_pos = data0[-1] # data0 = head data
        if self.prev_pos is None:
            self.prev_pos = cur_pos
            return 0 # no reward
        dist = self.prev_pos - cur_pos # distance moved
        print(f'distance moved : {dist}')
        self.prev_pos = cur_pos # set the prev dist to current value
        if dist and 10 < abs(dist) < 35: # ignore noise of abs(noise) > 10
            return dist*10 # scaled dist
        elif abs(dist) > 100:
            return -200 # direction changed, big penalty (expect it to restore)
        else:
            return -1 # constant penalty
        
    # receive next state from serial (controller)
    def step(self, state, action, term_flag = False):
        global FAIL, PROG, SUCCESS
        if isinstance(self.comm_obj, SerialComm):
            
            '''
                {'command' : 'action',
                'action1' : adc1,
                'action2' : adc2
                }
            '''
            
            # adc1 : head, adc2 : tail
            #print(action)
            adc1, adc2 = action # 거리증분치
            if not term_flag:
                msg = bytes(json.dumps({'command' : 'action',
                                        'action1' : int(adc1),
                                        'action2': int(adc2)}), encoding='utf-8')
            
            
            else: # flag to terminate current training session
                msg = bytes(json.dumps({'command' : 'end'}), encoding = 'utf-8')
            
            d = self.exchange_serial_msg(msg)
            
            
            if type(d) != dict or ('tag' not in d.keys()):
                # wrong message was received. terminate the training session.
                print(f'wrong message : {d}')
                return [None, None, None, None, None]
            
            if d['tag'] == 'end_serial':
                print(f'end message : {d}')
                return [None, None, None, None, None] # makes train() return -1
            
            if d['tag'] == 'next_state':         
                data = d['next_state']         
                         
                '''
                contents of d
                {'tag' : 'next_state',
                 'next_state' : recv_states -> dict}
                '''   
                
                # each of the entry containing 10-D value      
            
                # sorted device list (in order to keep the order)
                device_names = sorted(list(data.keys()))
                data_pair = [data[n] for n in device_names]
                state_lists = self.order_data_pair(data_pair) # sorted 10D(11D) vector
                state_lists_scaled = deepcopy(state_lists)
                # scale down some big values (adc, dist)
                for lst in state_lists_scaled:
                    if len(lst) == 10:
                        lst[-1] = lst[-1] / 1000 # divide adc val by 1000
                    if len(lst) == 11:
                        lst[-2] = lst[-2] / 1000
                        lst[-1] = lst[-1] / 100 # divide dist (mm) by 100
                
                
                state = state # not used  
                action = action 
                reward = self.return_reward(state_lists)
                print('reward : ', reward)
                next_state = state_lists_scaled[0] + state_lists_scaled[1] # concatenate (21-D)
                
                
                
                if reward == -1000: # fallen down, game over
                    status = FAIL
                elif state_lists[0][-1] < 40: 
                    status = SUCCESS # if dist < 35, assume we reached the terminal state
                else: # keep going, this episode has not ended yet
                    status = PROG
                
                
                self.comm_obj.dataCanvas.config(scrollregion=self.comm_obj.dataCanvas.bbox("all"))
                return [state, action, reward, next_state, status]
        
    # try to restore its initial states.  
    # send reset msg to the controller.  
    def reset(self):
        '''
        if isinstance(self.comm_obj, SerialComm):
            # ask the user to maually reset the state.
            
            reset_msg = bytes(json.loads({'tag' : 'reset'}))
            self.comm_obj.send()
            
            pass
        '''
        return [0. for _ in range(21)]
        

            
# DDPG
class Actor(Model):

    '''
    input : state
    output : action
    '''
    def __init__(self, action_dim, action_bound):
        super(Actor, self).__init__()

        self.action_bound = action_bound

        self.h1 = Dense(400, activation='relu')
        self.h2 = Dense(300, activation='relu',
                        bias_initializer=
                        tf.keras.initializers.random_uniform(
                            minval=-0.003, maxval=0.003))
        self.action = Dense(action_dim, activation='tanh')


    def call(self, state):
        x = self.h1(state)
        x = self.h2(x)
        a = self.action(x)

        return a


## critic network
class Critic(Model):

    '''
    input : [state, action]
    output : q value
    '''

    def __init__(self):
        super(Critic, self).__init__()

        self.h1 = Dense(400, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.L2(0.01))
        self.h2 = Dense(300, activation='relu',
                        bias_initializer=
                        tf.keras.initializers.random_uniform(
                            minval=-0.003, maxval=0.003),
                        kernel_regularizer=tf.keras.regularizers.L2(0.01))
        self.q = Dense(1, activation='linear')


    def call(self, state_action):
        state = state_action[0]
        action = state_action[1]
        xa = concatenate([state, action], axis=-1)
        x = self.h1(xa)
        x = self.h2(x)
        q = self.q(x)
        return q
    

## agent
class DDPGagent(object):

    def __init__(self):

        # hyperparameters
        self.GAMMA = 0.99
        self.BATCH_SIZE = 64
        self.BUFFER_SIZE = 1e6
        self.ACTOR_LEARNING_RATE = 0.001
        self.CRITIC_LEARNING_RATE = 0.01
        self.TAU = 0.1

        # serial communication
        self.comm_obj = SerialComm()
        self.comm_obj.initGui()
        self.env = Env(self.comm_obj)
        #print(self.comm_obj.serialObj.port)
        #print(self.comm_obj.serialObj.isOpen())
        
        # get state dimension
        self.state_dim = self.env.state_space_dim
        print('state dimension : ', self.state_dim)
        # get action dimension
        self.action_dim = self.env.action_space_dim
        print('action dimension : ', self.action_dim)
        # get action bound
        self.action_bound = self.env.action_bound
        print('action bound dim : ', self.action_bound)

        # create actor and critic networks
        self.actor = Actor(self.action_dim, self.action_bound)
        self.target_actor = Actor(self.action_dim, self.action_bound)

        self.critic = Critic()
        self.target_critic = Critic()

        self.actor.build(input_shape=(None, self.state_dim))
        self.target_actor.build(input_shape=(None, self.state_dim))

        state_in = Input((self.state_dim,))
        action_in = Input((self.action_dim,))
        self.critic([state_in, action_in])
        self.target_critic([state_in, action_in])

        self.actor.summary()
        self.critic.summary()

        # optimizer
        self.actor_opt = Adam(self.ACTOR_LEARNING_RATE)
        self.critic_opt = Adam(self.CRITIC_LEARNING_RATE)

        # initialize replay buffer
        self.buffer = ReplayBuffer(self.BUFFER_SIZE)

        # save the results
        self.save_epi_reward = []

    ## transfer actor weights to target actor with a tau
    def update_target_network(self, TAU):
        theta = self.actor.get_weights()
        target_theta = self.target_actor.get_weights()
        for i in range(len(theta)):
            target_theta[i] = TAU * theta[i] + (1 - TAU) * target_theta[i]
        self.target_actor.set_weights(target_theta)

        phi = self.critic.get_weights()
        target_phi = self.target_critic.get_weights()
        for i in range(len(phi)):
            target_phi[i] = TAU * phi[i] + (1 - TAU) * target_phi[i]
        self.target_critic.set_weights(target_phi)


    ## single gradient update on a single batch data
    def critic_learn(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            q = self.critic([states, actions], training=True)
            loss = tf.reduce_mean(tf.square(q-td_targets))

        grads = tape.gradient(loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(grads, self.critic.trainable_variables))


    ## train the actor network
    def actor_learn(self, states):
        with tf.GradientTape() as tape:
            actions = self.actor(states, training=True)
            critic_q = self.critic([states, actions])
            loss = -tf.reduce_mean(critic_q)

        grads = tape.gradient(loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(grads, self.actor.trainable_variables))


    ## computing TD target: y_k = r_k + gamma*Q(x_k+1, u_k+1)
    def td_target(self, rewards, q_values, dones):
        y_k = np.asarray(q_values)
        for i in range(q_values.shape[0]): # number of batch
            if dones[i]:
                y_k[i] = rewards[i]
            else:
                y_k[i] = rewards[i] + self.GAMMA * q_values[i]
        return y_k


    ## load actor weights
    def load_weights(self, path):
        self.actor.load_weights(path + 'walker_actor.h5')
        self.critic.load_weights(path + 'walker_critic.h5')    
    
    
    ## train the agent
    def train(self, max_episode_num):

        global FAIL, PROG, SUCCESS

        # initial transfer model weights to target model network
        self.update_target_network(1.0)
        term_flag = False # flag for termination of the training session
    
        # while a queen is not connected to this PC, block
        if isinstance(self.comm_obj, SerialComm):
            while not self.comm_obj.selectedPort:
                self.comm_obj.root.update()
                self.comm_obj.dataCanvas.config(scrollregion=self.comm_obj.dataCanvas.bbox("all"))
          
        for ep in range(int(max_episode_num)):
            # reset episode
            #print(self.comm_obj.serialObj.isOpen())
            time_counter, episode_reward, status = 0, 0, PROG
            
            # reset the env
            # THIS HAS TO BE DONE MANUALLY LATER (with distance sensor)
            self.env.prev_pos = None
            state = self.env.reset() #wrong
            pbar = tqdm(total=1000)

            while status == PROG:
                # if port closed, end training
                if self.comm_obj.end_flags[self.comm_obj.selectedPort]:
                    self.comm_obj.end_flags[self.comm_obj.selectedPort] = False
                    self.comm_obj.selectedPort = None
                    term_flag = True
                    
                # pick an action: shape = (2,)
                state_copy = deepcopy(state) # before normalization (for the step())
                state = self.env.normalize_state_vector(state) # np array [-1, 1]
                action = self.actor(tf.convert_to_tensor(np.array(state[np.newaxis, :]), dtype=tf.float32)) # (1, 11)
                #print('ACTION SHAPE : ',action.shape)
                action = action.numpy()[0]
                noise = np.random.randn(self.action_dim) * 0.1
                # clip continuous action to be within action_bound
                action = np.clip((action + noise)*self.action_bound, -self.action_bound, self.action_bound)
                print(action)

                # observe reward, new_state
                if not term_flag:
                    _, action, reward, next_state, status = self.env.step(state_copy, action)
                else:
                    _, action, reward, next_state, status = self.env.step(state_copy, action, term_flag = True)
                self.comm_obj.root.update()
                all_feedback_none = (action is None and\
                               reward is None and\
                               next_state is None and status is None)
                if all_feedback_none:
                    return -1 # terminate program
            
                #trained_reward = reward * 10.0 #??
                #self.buffer.add_buffer(state, action, trained_reward, next_state, done)

                self.buffer.add_buffer(state, action, reward, next_state, status)
                print(f'buffer count : {self.buffer.buffer_count()}')
                if self.buffer.buffer_count() <= 128:
                    pbar.update(1)

                if self.buffer.buffer_count() > 128:  # start train after buffer has some amounts

                    # sample transitions from replay buffer
                    states, actions, rewards, next_states, statuses = self.buffer.sample_batch(self.BATCH_SIZE)
                    #print(np.array([arr[0] for arr in next_states]).shape)
                    #print('aaaa : ', self.target_actor(tf.convert_to_tensor(np.array([arr[0] for arr in next_states]), dtype=tf.float32)))
                    #print('ggg : ', actions)    
                    # predict target Q-values
                    print(next_states.shape)
                    target_qs = self.target_critic([tf.convert_to_tensor(np.array(next_states), dtype=tf.float32),
                                                    self.target_actor(
                                                        tf.convert_to_tensor(next_states, dtype=tf.float32))])
                    # compute TD targets
                    y_i = self.td_target(rewards, target_qs.numpy(), statuses)

                    # train critic using sampled batch
                    self.critic_learn(tf.convert_to_tensor(next_states, dtype=tf.float32),
                                      tf.convert_to_tensor(actions, dtype=tf.float32),
                                      tf.convert_to_tensor(y_i, dtype=tf.float32))

                    # train actor
                    self.actor_learn(tf.convert_to_tensor(next_states, dtype=tf.float32))
                    # update both target network
                    #start_t = time.time()
                    self.update_target_network(self.TAU)
                    #end_t = time.time()
                    #print('gd time : ', end_t - start_t)

                # update current state
                state = next_state
                episode_reward += reward
                time_counter += 1
                self.comm_obj.dataCanvas.config(scrollregion=self.comm_obj.dataCanvas.bbox("all"))
                
                if episode_reward <= -1000:
                    break
                #print('state : ', state)
                
            # display rewards every episode
            print('Episode: ', ep+1, 'Time: ', time_counter, 'Reward: ', episode_reward)
            
            # block until a key comes in if the episode has ended
            key = input('episode end, continue ? (place robot at its initial state)')
            
            self.save_epi_reward.append(episode_reward)


            # save weights every episode
            if ep % 10 == 0:
                self.actor.save_weights("./save_weights/walker_actor.h5")
                self.critic.save_weights("./save_weights/walker_critic.h5")

            self.comm_obj.dataCanvas.config(scrollregion=self.comm_obj.dataCanvas.bbox("all"))

        np.savetxt('./save_weights/walker_epi_reward.txt', self.save_epi_reward)
        print(self.save_epi_reward)
        return 0 # end successfully


    ## save them to file if done
    def plot_result(self):
        plt.plot(self.save_epi_reward)
        plt.show()
    