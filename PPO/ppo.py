import numpy as np
import network
import json
from keras.optimizers import Adam
import tensorflow as tf
import logging
from dm_control.rl.control import Environment
from mujoco_viewer import Observer
import time
import tensorflow_probability as tfp
tfd = tfp.distributions

'''
for the reindeer env (수정가능)
'''
_state_dim = 8
_act_dim = 2


def init_hparams():
    with open('hparams.json', 'r') as f:
        fstr = f.read()
    if fstr:
        hparams_dict = json.loads(fstr)
        return hparams_dict

class PPO:
    def __init__(self, env):
        
        # 시각화를 위한 initialization (렉이 심해서 꺼놓았음)
        #self.observer = Observer(env=env, width=640, height=480, name='PPO')
        
        self.hparams = init_hparams()
        self.env = env
        self.state_dim = _state_dim
        self.act_dim = _act_dim
        
        # actor, critic MLP
        # actor : 미분가능한 state-policy parameterization
        # critic : 미분가능한 state-value parameterization
        self.actor = network.MLP(self.state_dim, self.act_dim)
        self.critic = network.MLP(self.state_dim, 1)
        
        # optimizers
        self.actor_optim = Adam(learning_rate=self.hparams['lr'])
        self.critic_optim = Adam(learning_rate=self.hparams['lr'])
        
        self.covar = tf.fill((self.act_dim,), 0.5) # [0.5, 0.5]
        self.covmat = tf.linalg.diag(self.covar)
        
        # 디버깅
        self.logger = {
			't_so_far': 0,          # timesteps so far
			'i_so_far': 0,          # iterations so far
			'batch_lens': [],       # episodic lengths in batch
			'batch_rews': [],       # episodic rewards in batch
			'actor_losses': [],     # losses of actor network in current iteration
		}
      
    def collect_batch(self):
        # actor가 T timestep (4800) 동안의 data를 collect하는 과정이다.
        
        assert isinstance(self.env, Environment)
        
        t = 0
        batch_s = []; batch_a = []; batch_log_probs = []
        batch_rews = []; batch_Q = []; batch_lens = []
        
        # 정해진 시간 (timesteps_per_batch) 동안 simulation
        while t < self.hparams['timesteps_per_batch']:
            ep_rews = [] # 에피소드마다 리턴된 rewards
            #print('start')
            timestep = self.env.reset()
            obs = timestep.observation # 맨 처음 state 관측치
            # obs는 nparray이므로 tf tensor로 바꿔준다.
            obs = tf.convert_to_tensor(obs, dtype=tf.float32)
            
            # 에피소드 한 번 simulation
            for ep_t in range(self.hparams['max_timesteps_per_episode']):
                #print(ep_t)
                batch_s.append(obs)
                # actor network의 policy distribution을 평균값으로 하는
                # 다변수정규분포에서 action 샘플링
                action, log_prob = self.sample_from_dist(obs)
                timestep = self.env.step(action)
                
                '''
                # 시각화 뷰어에 step 이벤트 발생 (무시해도 됨)
                self.observer.step(0, None, None)
                self.observer._render()
                '''
                
                ep_rews.append(timestep.reward) # reward
                batch_a.append(action) # action
                batch_log_probs.append(log_prob) # pi(a|s)
                
                if timestep.last(): # 만약에 이 episode가 중단됐다면
                    break
                
                t += 1 # timestep을 1만큼 증가
                
            batch_lens.append(ep_t + 1) # 나중에 batch 하나가 만들어지는데 timestep이 얼마나 걸렸는지 확인
            batch_rews.append(ep_rews) # timestep마다 얻은 reward 기록
            
        # 샘플링 된 배치 (리스트) 들을 tf tensor로 변환
        batch_s = tf.convert_to_tensor(batch_s, dtype=tf.float32) # shape : (T, 9)
        print(f'batch_s : {batch_s}')
        batch_a = tf.convert_to_tensor(batch_a, dtype=tf.float32) # shape : (T, 2)
        batch_log_probs = tf.convert_to_tensor(batch_log_probs, dtype=tf.float32) # shape : (T,)
        
        # batch_rews 값을 바탕으로 batch_Q 값 계산 (discounted rewards)
        for ep_rews in reversed(batch_rews):
            discounted_rew = 0
            # total discounted rewards 계산, episode별로 기록
            for rew in reversed(ep_rews): # 반대 방향으로 계산
                # Q' = R + gamma*Q 
                discounted_rew = rew + self.hparams['gamma']*discounted_rew
                batch_Q.insert(0, discounted_rew) # 반대 방향으로 기록
        
        batch_Q = tf.convert_to_tensor(batch_Q, dtype=tf.float32) # shape : (T,)
        
        # 디버깅
        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens
                
        return batch_s, batch_a, batch_log_probs, batch_Q, batch_lens
    
    
    # action a와 그 a에 대한 pi(a|s)를 반환        
    def sample_from_dist(self, state_batch=None):
        state_batch = state_batch[tf.newaxis, :] 
        mean = self.actor(state_batch)
        # actor의 출력값을 평균으로 갖고, 0.5의 공분산을 갖는 정규분포에서 action 샘플링
        dist = tfd.MultivariateNormalFullCovariance(mean, self.covmat)
        action_batch = dist.sample()
        
        # pi_theta(a|s)를 log 취한 값
        log_probs = dist.log_prob(action_batch)
        
        return np.array(action_batch), np.array(log_probs)         
        
    # state가 주어졌을 때 해당 state batch에 대한 value batch(예측값)반환 
    # return type : tf tensor
    def compute_value(self, state_batch=None):
        #print(f'state batch in compute_value : {tf.shape(state_batch)}')
        return self.critic(state_batch)
    
    # state가 주어졌을 때 해당 action batch에 대한 policy(a,s) batch 반환
    # return type : tf tensor
    def compute_pi(self, state_batch=None, action_batch=None):
        state_batch = state_batch[tf.newaxis, :] 
        mean = self.actor(state_batch)
        dist = tfd.MultivariateNormalFullCovariance(mean, self.covmat)
        return dist.log_prob(action_batch) # policy(A=a|S=s)
    
    def print_log(self):
        # 디버깅용
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        
        avg_actor_loss = np.mean(np.array(self.logger['actor_losses']))

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

		# Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Reward: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []
        
    def learn(self, T):
        # 전체 학습 시간 : T timesteps
        
        # 시각화를 위한 설정
        #self.observer.begin_episode()
        # logger timer 시작
        self.logger['delta_t'] = time.time_ns()
    
        
        t_so_far = 0; i_so_far = 0
        while t_so_far < T: 
                
            # GradientTape로 gradient graph를 계산한다.    
            with tf.GradientTape(persistent=True) as tape_actor, tf.GradientTape(persistent=True) as tape_critic:
                # run policy pi_theta_old in env for T timesteps (논문)
                batch_s, batch_a, batch_log_probs, batch_Q, batch_lens = self.collect_batch()
                # 4800 개 (timestep) 동안의 데이터를 가지고 있다.
                
                # batch sampling timestep만큼 지남
                t_so_far += np.sum(batch_lens)
                
                self.logger['t_so_far'] = t_so_far
                
                # 현재 iteration 에서의 Advantage function estimate 계산 (논문)
                V = self.compute_value(state_batch=batch_s) # scalar
                #print(f'batch_Q : {batch_Q}, V : {V}')
                A_i = batch_Q - V # shape : (T,)
                
                # 추가 (학습 안정성)
                A_i = (A_i - tf.math.reduce_mean(A_i)) / (tf.math.reduce_std(A_i) + 1e-10)
                
                # epoch = n_updates_per_iteration 동안 parameter update
                # 본격적으로 학습하는 부분
                
                print('start policy gradient..')
                
                for _ in range(self.hparams['n_updates_per_iteration']):
                    V = self.compute_value(batch_s) # V_pi (prediction)
                    cur_log_probs = self.compute_pi(batch_s, batch_a) # pi_theta(a_t | s_t)
                    
                    # batch_log_probs -> 이전의 policy (pi_theta_old(a_t | s_t))
                    # cur_log_probs -> 현재의 policy (pi_theta(a_t | s_t))
                    log_ratios = cur_log_probs - batch_log_probs 
                    ratios = tf.exp(log_ratios) # r_t(theta) -> shape: (T,)
                    
                    #print(f'RATIOS : {ratios}')
                    
                    _epsilon = self.hparams['clip_epsilon']
                    # surrogate L_clip (논문)
                    L_clip = tf.minimum(
                        ratios * A_i,
                        tf.clip_by_value(ratios, 1.-_epsilon, 1.+_epsilon) * A_i                  
                    )
                    
                    #print(f'TERM 1 : {ratios * A_i}')
                    #print(f'TERM 2 : {tf.clip_by_value(ratios, 1.-_epsilon, 1.+_epsilon) * A_i}')
                    
                    #print(f'L_CLIP : {L_clip}')
                    
                    np.save('./numpy.npy', -L_clip)
                    
                    
                    #print(f'ACTOR TRAINABLE : {self.actor.trainable_variables}')
                    #print(f'CRITIC TRAINABLE : {self.critic.trainable_variables}')
                    
                    actor_loss = tf.math.reduce_mean(-L_clip)
                    print(f"actor_loss : {actor_loss}")
                    
                    actor_grads = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
                    #print(f'actor grads : {actor_grads}')
                    self.actor_optim.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
                    
                    
                    critic_loss = tf.losses.mean_squared_error(V, batch_Q)
                    
                    critic_grads = tape_critic.gradient(critic_loss, self.critic.trainable_variables)
                    #print(f'critic grads : {critic_grads}')
                    self.critic_optim.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
                    
                self.logger['actor_losses'].append(actor_loss)
            
            
            self.print_log()
            i_so_far += 1
            
            # save_freq 주기마다 한번씩 weight를 저장한다. (checkpoint)
            if i_so_far % self.hparams['model_save_freq'] == 0:
                self.actor.save(save_format='tf', filepath='./weights/actor')
                self.critic.save(save_format='tf', filepath='./weights/critic')
            
                    
            