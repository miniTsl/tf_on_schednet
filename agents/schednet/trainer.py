from __future__ import print_function, division, absolute_import
from time import time

import numpy as np
import logging

from agents.schednet.agent import PredatorAgent
from agents.evaluation import Evaluation

from envs.gui import canvas

# 获取命令行参数！！！
import config
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
        self._env = env # 一个wrapper对象
        self._eval = Evaluation()
        self._agent_profile = env.env.agent_profile # 想办法换掉这个东西
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
        
        # State and obs <additionally include <history information> >
        self._state_dim = self._n_predator*2 + self._n_predator 
        # 本质上state_dim就是num_agent * 2(x,y) + schedule数量（即n_agent），这点在PP和TF中都是一样的
        self._obs_dim = 62 # 这里手动设置下TF环境中的观测值的维度，如何让他自动计算？？？？
        
        # get predator agent
        self._predator_agent = PredatorAgent(n_agent=self._agent_profile['predator']['n_agent'],
                                             action_dim=self._agent_profile['predator']['act_dim'], # 1？
                                             state_dim=self._state_dim, # num_agent * 2(x,y) + schedule数量（即n_agent）
                                             obs_dim=self._obs_dim) # 62

        self.epsilon = 0.5  # Init value for epsilon

        if FLAGS.gui:  # Enable GUI
            self.canvas = canvas.Canvas(self._n_predator, 1, FLAGS.map_size)
            self.canvas.setup()

    def learn(self):
        global_step = 0 # 总步数
        epochs = 0
        while epochs < num_epochs:
            # 新epoch
            epochs += 1
            step_in_epoch = 0
            episode_num = 0 # 这个epoch内的总轨迹数
            total_reward = np.zeros(self._n_predator)    # 这个epoch内总的reward，一个n_predator大小的ndarray
            success_num = 0 # 这个epoch内成功的episode数
            epoch_strat_time = time()
            while step_in_epoch < epoch_size:   
                # 新轨迹
                step_in_episode = 0  # 本轨迹内用的总步数
                episode_num += 1    # 轨迹数+1
                done = False    # 本轨迹未完
                
                obs_n = self._env.reset(epochs-1)   
                # 本轨迹通过envwrapper而初始环境,注意epochs需要-1，为了与IC3net中envwrapper保持一致
                # ic3net环境reset返回的obs中的每一个agent的元素是（tuple(tuple（数, 数, array（1，1，59）)）,而且最里层的三个元素的形状还不一样，在envwrapper中每一个大tuple都转换成了一个list
                
                
                info_n = self._env.env.get_state()   
                # PP 中：
                #     info_n是一个字典list，每个字典键值对为"state": 行array        list中共n_agent个字典。其实只有第一个有用，因为它们都一样的
                #     state行array的内容是map中 <所有agent的方位信息> 即全局信息，大小为 agent_num*2(x,y)
                # TF中:
                #     info_n是一个字典，其中的car_loc对应的就是map中所有车的位置信息，就是一个二维array，n_agent行2列
                
                '''# 注意此时的obs和info_n的最后还没有添加schedule'''
                  
                  
                h_schedule_n = np.zeros(self._n_predator)  # schedule history，类型是 np.array，包含每个agent的调度值                
                # 在obs和state后面分别添加历史信息，正如 第58行所说。同时将obs的数据类型修改为ndarray
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
                    # 这时obs_n已经是ndarray了
                    schedule_n, priority = self.get_schedule(obs_n, global_step, FLAGS.sched)
                    action_n = self.get_action(obs_n, schedule_n, global_step)
                    obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
                    '''从PP环境中返回的obs_n、reward、done、info都是list。
                    ic3net中env返回的obs是（tuple(tuple（数, 数, array（1，1，59）)）,而且三个array的形状还不一样，在envwrapper中转换成了list
                    reward_n其实不必由ic3net的ndarray转换为list因为trainer.py中反正要求和并且转换为nparray；
                    done：反正PP代码中也要对所有的done求和，干脆返回TF中的episode_over，一个bool类型的数'''
                    
                    obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)
                    '''
                    schednet中增加历史信息后的state是ndarray，里面有14个元素（2*5+4）
                    reward是list，里面有5个数，是每个agent获得的奖励值
                    priority是ndarray，里面有4个数，是agent的调度概率，schedule_n是调度值（0 or 1）
                    done_n是False或者True，done_single一样 
                    '''
                    done_single = done_n    # bool类型
                    
                    # 传入train_agents中的都是ndarray
                    self.train_agents(state, obs_n, action_n, reward_n, state_next, obs_n_next, schedule_n, priority, done_single, global_step)
                    
                    # 更新！！！
                    obs_n = obs_n_next
                    state = state_next
                    total_reward += reward_n    # 整个epoch内每一步都要叠加

                    done = is_episode_done(done_n, step_in_episode)
                    if done:    # 这个episode结束了 
                        success_num +=  (1 - self._env.env.has_failed)  # 这个episode是否算是成功的
            # 打印这个epoch的信息
            np.set_printoptions(precision=2)
            print("[Train_epoch %d]\n" % (epochs),"Total_steps_till_now:", global_step, " Success_rate: {:.2f}".format(success_num/episode_num),
                  " Time: {:.2f}s".format(time()-epoch_strat_time), " Add_rate: {:.2f}".format(self._env.env.add_rate) , "\n", "Ave_reward:", total_reward/episode_num )

            if FLAGS.eval_on_train and global_step % FLAGS.eval_step == 0:
                self.test(epochs)
            
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
            predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]   # list of array
            predator_action = self._predator_agent.act(predator_obs, schedule_n)

        for i, idx in enumerate(self._agent_profile['predator']['idx']):
            act_n[idx] = predator_action[i]
        
        # TFJ不需要猎物，所以没有相应的action
        
        # 将action转换为np array
        return np.array(act_n, dtype=np.int32)

    def get_schedule(self, obs_n, global_step, type, train=True):
        '''
        return:
            ret: array [ _n_predator ] of 0 or 1
            priority: weight array for schedule. used for backward learning
        '''
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]   # list of array

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

    def train_agents(self, state, obs_n, action_n, reward_n, state_next, obs_n_next, schedule_n, priority, done, global_step):
        predator_obs = [obs_n[i] for i in self._agent_profile['predator']['idx']]
        predator_action = [action_n[i] for i in self._agent_profile['predator']['idx']]
        predator_reward = [reward_n[i] for i in self._agent_profile['predator']['idx']]
        predator_obs_next = [obs_n_next[i] for i in self._agent_profile['predator']['idx']]
        self._predator_agent.train(state, predator_obs, predator_action, predator_reward,
                                   state_next, predator_obs_next, schedule_n, priority, done, global_step)

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
            这个函数返回修改后的obs、state、schedule_n，增加了调度信息(这一step产生的schedule和旧schedule的滑动平均)
            修改schedule的目的是为了更好地进行训练，不是为了决策
        '''
        if not init:    
            h_schedule_n = self.update_h_schedule(h_schedule_n, schedule_n) # 滑动平均
        
        # 希望把历史的h_schedule_n转换为列向量补充在obs_n_ws的最后一列
        obs_n_h = np.concatenate((obs_n_ws[0:self._n_predator], h_schedule_n.reshape((self._n_predator,1))), axis=1)
        
        obs_final = list()
        for i in range(self._n_predator):
            obs_final.append(obs_n_h[i])

        '''再把array的list转变为二维array'''
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
    
    def test(self, epochs=None):
        global_step = 0
        episode_num = 0
        total_reward = 0
        success_num = 0
        
        test_strat_time = time()
        while global_step < testing_step:
            done = False
            episode_num += 1
            step_in_episode = 0
            obs_n = self._env.reset(epochs)  
            info_n = self._env.env.get_state()
            h_schedule_n = np.zeros(self._n_predator)
            obs_n, state, _ = self.get_obs_state_with_schedule(obs_n, info_n, h_schedule_n, init=True)

            while not done:
                global_step += 1
                step_in_episode += 1

                schedule_n, priority = self.get_schedule(obs_n, global_step, FLAGS.sched)
                action_n = self.get_action(obs_n, schedule_n, global_step, False)
                obs_n_without_schedule, reward_n, done_n, info_n = self._env.step(action_n)
                obs_n_next, state_next, h_schedule_n = self.get_obs_state_with_schedule(obs_n_without_schedule, info_n, h_schedule_n, schedule_n)

                obs_n = obs_n_next
                state = state_next
                total_reward += reward_n

                done = is_episode_done(done_n, step_in_episode)#, "test"):
                if done:
                    success_num += (1-self._env.env.has_failed)
        np.set_printoptions(precision=2)
        print("[Test_after_epoch %d]\n" % (epochs)," Success_rate: {:.2f}".format(success_num/episode_num),
                " Time: {:.2f}s".format(time()-test_strat_time), " Add_rate: {:.2f}".format(self._env.env.add_rate) , "\n", "Ave_reward:", total_reward/episode_num )

def is_episode_done(done, step):
    if done or step >= FLAGS.max_steps:
        return True
    else:
        return False

