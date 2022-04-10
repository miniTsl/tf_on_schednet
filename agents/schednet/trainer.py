from __future__ import print_function, division, absolute_import
from time import time

import numpy as np
import logging

from agents.schednet.agent import PredatorAgent
from agents.simple_agent import RandomAgent
from agents.evaluation import Evaluation

import config
from envs.gui import canvas

FLAGS = config.flags.FLAGS
logger = logging.getLogger('Agent')
result = logging.getLogger('Result')

training_step = FLAGS.training_step 
testing_step = FLAGS.testing_step   
num_epochs = FLAGS.num_epochs
epoch_size = FLAGS.epoch_size

epsilon_dec = 1.0/training_step
epsilon_min = 0.1


class Trainer(object):

    def __init__(self, env):
        logger.info("SchedNet trainer is created")
        self._env = env
        self._eval = Evaluation()
        self._agent_profile = env.env.agent_profile
        '''
        PP环境下的：
        _agent_profile:{'predator': {'n_agent': 4, 'idx': [...], 'act_dim': 5, 'com_dim': 0, 'obs_dim': (...)}, 'prey': {'n_agent': 1, 'idx': [...], 'act_dim': 5, 'com_dim': 0, 'obs_dim': (...)}}
            special variables
            function variables
            'predator':{'n_agent': 4, 'idx': [0, 1, 2, 3], 'act_dim': 5, 'com_dim': 0, 'obs_dim': (6,)}
                special variables
                function variables
                'n_agent':4
                'idx':[0, 1, 2, 3]
                'act_dim':5
                'com_dim':0
                'obs_dim':(6,)
                len():5
            'prey':{'n_agent': 1, 'idx': [4], 'act_dim': 5, 'com_dim': 0, 'obs_dim': (2,)}
                special variables
                function variables
                'n_agent':1
                'idx':[4]         
                'act_dim':5
                'com_dim':0
                'obs_dim':(2,)
                len():5
        '''
        self._n_predator = self._agent_profile['predator']['n_agent']
        #self._n_prey = self._agent_profile['prey']['n_agent']
        
        # State and obs <additionally include <history information>
        # PP中的state_dim包括了调度信息（调度共n_predator个）
        self._state_dim = self._n_predator*2 + self._n_predator # 本质上state_dim就是num_agent * 2(x,y) + schedule数量（即n_agent）

        self._obs_dim = 62#self._agent_profile['predator']['obs_dim'][0] + 1
        
        # get predator agent
        self._predator_agent = PredatorAgent(n_agent=self._agent_profile['predator']['n_agent'],
                                             action_dim=self._agent_profile['predator']['act_dim'],
                                             state_dim=self._state_dim,
                                             obs_dim=self._obs_dim)
        # Prey agent (randomly moving)
        '''
        self._prey_agent = []
        for _ in range(self._n_prey):
            self._prey_agent.append(RandomAgent(5))
        '''
        self.epsilon = 0.5  # Init value for epsilon

        if FLAGS.gui:  # Enable GUI
            self.canvas = canvas.Canvas(self._n_predator, 1, FLAGS.map_size)
            self.canvas.setup()

    def learn(self):
        global_step = 0 # 总步数
        print_flag = True   # 打印训练数据
        epochs = 0
        while epochs < num_epochs:
            # 新epoch
            epochs += 1
            step_in_epoch = 0
            episode_num = 0 # 这个epoch内的总轨迹数
            total_reward = np.zeros(self._n_predator)    # 这个epoch内总的reward，一个n_predator大小的容器
            success_num = 0 # 这个epoch内成功的次数
            epoch_strat_time = time()
            while step_in_epoch < epoch_size:   
                # 新轨迹
                step_in_episode = 0  # 本轨迹内用的总步数
                episode_num += 1    # 轨迹数+1
                done = False    # 本轨迹未完
                obs_n = self._env.reset(1)   # 本轨迹通过envwrapper而初始环境
                '''从PP环境中返回的obs_n、reward、done、info都是list。
                ic3net返回的obs是（tuple(tuple（数, 数, array（1，1，59）)）,而且三个array的形状还不一样，在envwrapper中转换成了list'''
                
                info_n = self._env.env.get_state()   
                '''PP 中：
                # info_n是一个字典list，每个字典键值对为state: 行array        list中共n_agent个字典。其实只有第一个有用，因为它们都一样的
                # state行array的内容是图中 <所有agent的方位信息>即全局信息，大小为 agent_num*2(x,y)
                '''
                
                h_schedule_n = np.zeros(self._n_predator)  # schedule history
                # 类型是 np.array，包含每个agent的调度值
                
                # 在obs和state后面分别添加历史信息，正如 第58行所说，同时将obs的数据类型修改为ndarray
                obs_n, state, _ = self.get_obs_state_with_schedule(obs_n, info_n, h_schedule_n, init=True)
                ''' <schednet要求的最后的obs_n的数据类型是二维array> 
                    pp环境的obs(最后一列已经加了调度值)最开始的一次观测：
                    0:array([0.66666667, 0.83333333, 1.        , 1.        , 0.        ,0.5       , 0.        ])
                    1:array([0.66666667, 1.        , 0.        , 0.        , 0.5       ,0.5       , 0.        ])
                    2:array([0.16666667, 0.33333333, 0.        , 0.        , 0.5       ,0.5       , 0.        ])
                    3:array([0.83333333, 0.33333333, 1.        , 1.        , 1.        ,0.        , 0.        ])
                    4:array([0.66666667, 0.5       ])
                '''
                while not done: # 本轨迹未结束
                    global_step += 1
                    step_in_epoch += 1
                    step_in_episode += 1

                    schedule_n, priority = self.get_schedule(obs_n, global_step, FLAGS.sched)
                    action_n = self.get_action(obs_n, schedule_n, global_step)
                    obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
                    '''从PP环境中返回的obs_n、reward、done、info都是list。
                    ic3net返回的obs是（tuple(tuple（数, 数, array（1，1，59）)）,而且三个array的形状还不一样，在envwrapper中转换成了list
                    reward_n其实不必由ndarray转换为list因为train中反正要求和，done反正PP代码中也要对所有的done求和，干脆返回TF中的episode_over，一个bool类型的数'''
                    # reward_n = reward_n.tolist()
                    
                    obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)
                    done_single = done_n    # bool类型即可
                    '''
                    schednet中state是ndarray，里面有14个元素（2*5+4），又忘了？是全局的方位信息！
                    reward是list，里面有5个数，是每个agent获得的奖励值
                    priority是ndarray，里面有4个数，应该是agent的调度权重？
                    done_n是list，里面是False或者True，done_single是bool类型 
                    '''
                    
                    # 传入train_agents中的都是ndarray
                    self.train_agents(state, obs_n, action_n, reward_n, state_next, obs_n_next, schedule_n, priority, done_single)
                    
                    if FLAGS.gui:
                        self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train")
                    
                    # 更新！！！
                    obs_n = obs_n_next
                    state = state_next
                    total_reward += reward_n

                    done = is_episode_done(done_n, step_in_episode)
                    if done:
                        success_num +=  (1-self._env.env.has_failed)
                    # if is_episode_done(done_n, step_in_ep):    # 应该再加个step_in_ep<500？还是说训练时每一条轨迹一口气走到黑
                    #     if FLAGS.gui:
                    #         self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train", True)
                    #     if print_flag:
                    #         print("[train_episode %d]" % (episode_num)," total step till now:", global_step, " step_used:", step_in_ep, " reward", total_reward)
                    #     done = True

                    # if FLAGS.eval_on_train and global_step % FLAGS.eval_step == 0:
                    #     self.test(global_step)
                    #     break
            np.set_printoptions(precision=3)
            print("[Train_epoch %d]\n" % (epochs),"Total_steps_till_now:", global_step, " Success_rate: {:.3f}".format(success_num/episode_num),
                  " Time: {:.2f}s".format(time()-epoch_strat_time), " Add_rate: {:.4f}".format(self._env.env.add_rate) , "\n", "Epoch_average_reward: ", total_reward/episode_num )
        
            
        
        
        
        
        
        
        
        
        
        
        
        # while global_step < training_step:
        #     # 新的轨迹  
        #     episode_num += 1
        #     step_in_ep = 0  # 本轨迹内用的总步数
        #     total_reward = 0
        #     done = False
        #     obs_n = self._env.reset(1)   # envwrapper中的初始环境
        #     '''从PP环境中返回的obs_n、reward、done、info都是list。
        #     ic3net返回的obs是（tuple(tuple（数, 数, array（1，1，59）)）,而且三个array的形状还不一样，在envwrapper中转换成了list'''
            
        #     info_n = self._env.env.get_state()   
        #     '''PP 中：
        #     # info_n是一个字典list，每个字典键值对为state: 行array        list中共n_agent个字典。其实只有第一个有用，因为它们都一样的
        #     # state行array的内容是图中 <所有agent的方位信息>即全局信息，大小为 agent_num*2(x,y)
        #     '''
            
        #     h_schedule_n = np.zeros(self._n_predator)  # schedule history
        #     # 类型是 np.array，包含每个agent的调度值
            
        #     # 在obs和state后面分别添加历史信息，正如 第58行所说，同时将obs的数据类型修改为ndarray
        #     obs_n, state, _ = self.get_obs_state_with_schedule(obs_n, info_n, h_schedule_n, init=True)
        #     ''' <schednet要求的最后的obs_n的数据类型是二维array> 
        #         pp环境的obs(最后一列已经加了调度值)最开始的一次观测：
        #         0:array([0.66666667, 0.83333333, 1.        , 1.        , 0.        ,0.5       , 0.        ])
        #         1:array([0.66666667, 1.        , 0.        , 0.        , 0.5       ,0.5       , 0.        ])
        #         2:array([0.16666667, 0.33333333, 0.        , 0.        , 0.5       ,0.5       , 0.        ])
        #         3:array([0.83333333, 0.33333333, 1.        , 1.        , 1.        ,0.        , 0.        ])
        #         4:array([0.66666667, 0.5       ])
        #     '''
            
        #     while not done: # 本轨迹未结束
        #         global_step += 1
        #         step_in_ep += 1

        #         schedule_n, priority = self.get_schedule(obs_n, global_step, FLAGS.sched)
                
        #         action_n = self.get_action(obs_n, schedule_n, global_step)
                
        #         obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
        #         '''从PP环境中返回的obs_n、reward、done、info都是list。
        #         ic3net返回的obs是（tuple(tuple（数, 数, array（1，1，59）)）,而且三个array的形状还不一样，在envwrapper中转换成了list
        #         reward_n其实不必由ndarray转换为list因为train中反正要求和，done反正PP代码中也要对所有的done求和，干脆返回TF中的episode_over，一个bool类型的数'''
        #         # reward_n = reward_n.tolist()
                
        #         obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)
        #         done_single = done_n    # bool类型即可
        #         '''
        #         schednet中state是ndarray，里面有14个元素（2*5+4），又忘了？是全局的方位信息！
        #         reward是list，里面有5个数，是每个agent获得的奖励值
        #         priority是ndarray，里面有4个数，应该是agent的调度权重？
        #         done_n是list，里面是False或者True，done_single是bool类型 
        #         '''
                
        #         # 传入train_agents中的都是ndarray
        #         self.train_agents(state, obs_n, action_n, reward_n, state_next, obs_n_next, schedule_n, priority, done_single)
                
        #         if FLAGS.gui:
        #             self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train")
                
        #         # 更新！！！
        #         obs_n = obs_n_next
        #         state = state_next
        #         total_reward += np.sum(reward_n)

        #         if is_episode_done(done_n, step_in_ep):    # 应该再加个step_in_ep<500？还是说训练时每一条轨迹一口气走到黑
        #             if FLAGS.gui:
        #                 self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Train", True)
        #             if print_flag:
        #                 print("[train_episode %d]" % (episode_num)," total step till now:", global_step, " step_used:", step_in_ep, " reward", total_reward)
        #             done = True

        #         if FLAGS.eval_on_train and global_step % FLAGS.eval_step == 0:
        #             self.test(global_step)
        #             break
        self._predator_agent.save_nn(global_step)
        self._eval.summarize()

    def get_action(self, obs_n, schedule_n, global_step, train=True):
        act_n = [0] * len(obs_n)
        
        # epsilon（用于探索新的action）迭代减小
        self.epsilon = max(self.epsilon - epsilon_dec, epsilon_min)
        
        # 需要把predator action修改为0，1两种，才可以用于环境的step
        # Action of predator
        if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):  # with prob. epsilon
            # Exploration探索
            predator_action = self._predator_agent.explore()
        else:
            # Exploitation利用已有策略
            predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
            predator_action = self._predator_agent.act(predator_obs, schedule_n)

        for i, idx in enumerate(self._agent_profile['predator']['idx']):
            act_n[idx] = predator_action[i]
        
        # TFJ不需要猎物，所以没有相应的action
        # Action of prey
        '''
        for i, idx in enumerate(self._agent_profile['prey']['idx']):
            act_n[idx] = self._prey_agent[i].act(None)
        '''
        
        # 将action转换为np array
        return np.array(act_n, dtype=np.int32)

    def get_schedule(self, obs_n, global_step, type, train=True):
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]

        if train and (global_step < FLAGS.m_size * FLAGS.pre_train_step or np.random.rand() < self.epsilon):
            # Exploration: Schedule k random agent
            priority = np.random.rand(self._n_predator)
            i = np.argsort(-priority)[:FLAGS.s_num]  
            ret = np.full(self._n_predator, 0.0)
            ret[i] = 1.0
            return ret, priority
        else:
            # Exploitation
            return self._predator_agent.schedule(predator_obs)

    def train_agents(self, state, obs_n, action_n, reward_n, state_next, obs_n_next, schedule_n, priority, done):
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        predator_obs_next = [obs_n_next[i] for i in self._agent_profile['predator']['idx']]
        self._predator_agent.train(state, predator_obs, predator_action, predator_reward,
                                   state_next, predator_obs_next, schedule_n, priority, done)

    def get_h_obs_state(self, obs_n, state, h_schedule):
        obs_n_h = np.concatenate((obs_n[0:self._n_predator], h_schedule.reshape((self._n_predator,1))), axis=1)
        obs_final = list()
        for i in range(self._n_predator):
            obs_final.append(obs_n_h[i])
        for i in range(self._n_prey):
            obs_final.append(obs_n[self._n_predator + i])
        obs_n = np.array(obs_final)
        state = np.concatenate((state, h_schedule), axis=-1)

        return obs_n, state

    def get_obs_state_with_schedule(self, obs_n_ws, info_n, h_schedule_n, schedule_n=None, init=False):
        '''
            这个函数的返回修改后的obs、state、schedule_n，增加了历史数据，但是有点奇怪，中间隔了一步的感觉
        '''
        # 运行过程中采用移动平均的办法更新调度值，进而用来训练。不是为了决策action
        if not init:    
            h_schedule_n = self.update_h_schedule(h_schedule_n, schedule_n)
        # 希望把历史的h_schedule_n转换为列向量补充在obs_n_ws的最后一列
        obs_n_h = np.concatenate((obs_n_ws[0:self._n_predator], h_schedule_n.reshape((self._n_predator,1))), axis=1)
        
        obs_final = list()
        for i in range(self._n_predator):
            obs_final.append(obs_n_h[i])

        # 最后再把array的list转变为一个array（size 5），每个元素又是一个array
        obs_n = np.array(obs_final)
        
        # 按照列往右去拼接~，state在原来info_n['car_loc']的基础上增加了n_predator个元素
        state = np.concatenate((info_n['car_loc'].flatten(), h_schedule_n), axis=-1)
        
        return obs_n, state, h_schedule_n

    def update_h_schedule(self, h_schedule, schedule_n):
        ret = h_schedule * 0.5 + schedule_n * 0.5
        return ret

    def print_obs(self, obs):
        for i in range(FLAGS.n_predator):
            print(obs[i])
        print("")

    def check_obs(self, obs):
        check_list = []
        for i in range(FLAGS.n_predator):
            check_list.append(obs[i][2])

        return np.array(check_list)
    
    def test(self, curr_ep=None):
        global_step = 0
        episode_num = 0
        total_reward = 0
        obs_cnt = np.zeros(self._n_predator)
        
        while global_step < testing_step:

            episode_num += 1
            step_in_ep = 0
            obs_n = self._env.reset()  
            info_n = self._env.env.get_info()
            h_schedule_n = np.zeros(self._n_predator)
            obs_n, state, _ = self.get_obs_state_with_schedule(obs_n, info_n, h_schedule_n, init=True)

            while True:
                global_step += 1
                step_in_ep += 1

                schedule_n, priority = self.get_schedule(obs_n, global_step, FLAGS.sched)
                action_n = self.get_action(obs_n, schedule_n, global_step, False)
                obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
                obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)

                obs_cnt += self.check_obs(obs_n_next)

                if FLAGS.gui:
                    self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Test")

                obs_n = obs_n_next
                state = state_next
                total_reward += np.sum(reward_n)

                if is_episode_done(done_n, global_step, "test") or step_in_ep > FLAGS.max_step:
                    if FLAGS.gui:
                        self.canvas.draw(state_next * FLAGS.map_size, [0]*self._n_predator, "Test", True)
                    break

        print("Test result at total steps: ", curr_ep, " Average steps to capture: ", float(global_step) / episode_num,
              " Average reward: ", float(total_reward) / episode_num, "  Average obs_cnt: ",obs_cnt / episode_num)
        self._eval.update_value("test_result", float(global_step)/episode_num, curr_ep) # 结果文件中保存test数据


def is_episode_done(done, step, e_type="train"):
    if e_type == "test":
        if done or step >= FLAGS.testing_step:
            return True
        else:
            return False

    else:
        if done or step >= FLAGS.max_steps:
            return True
        else:
            return False

