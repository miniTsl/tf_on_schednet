from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import itertools
import numpy as np

from envs.grid_core import World, CoreAgent
from envs.scenario import BaseScenario

import config

FLAGS = config.flags.FLAGS
IDX_TO_OBJECT = config.IDX_TO_OBJECT

# 猎物类
class Prey(CoreAgent):
    '''方法：is_captured(self, world)（猎物是否被抓住）--> bool'''
    def __init__(self):
        super(Prey, self).__init__('prey', 'green')
        self._movement_mask = np.array(
            [[0,1,0],
             [1,0,1],
             [0,1,0]], dtype=np.int8
        )

    def is_captured(self, world):
        x, y = self.pos
        minimap = world.grid.slice(x-1,y-1,3,3).encode()[:,:,0] != 0
        return np.sum(minimap*self._movement_mask) == 4

# 捕食者类
class Predator(CoreAgent):
    '''方法：set_obs_prey，reset_obs_prey，set_obs_prey'''
    def __init__(self, id=0):
        super(Predator, self).__init__('predator', 'blue')
        self.id = id

        if FLAGS.hetero > 0:  # Hetero捕食者属于异质性哦那，第一个捕食者的观测范围是5*5，其他都是3*3
            if self.id == 0:
                self.obs_range = 2
            else:
                self.obs_range = 1

        else:  # Homogeneous 捕食者属于同质
            self.obs_range = FLAGS.obs_range    # 默认是1，对应3*3

        self.last_obs_x = 0.5
        self.last_obs_y = 0.5
        # 之前是否看到过猎物
        self.obs_prey_before = False
        
    def set_obs_prey(self, px, py):
        '''设置之前看到的猎物的坐标'''
        self.last_obs_x = px
        self.last_obs_y = py
        self.obs_prey_before = True

    def reset_obs_prey(self):
        self.last_obs_x = 0.5
        self.last_obs_y = 0.5
        self.obs_prey_before = False

    def get_obs_prey(self):
        '''获取之前看到的猎物的情况、坐标x、坐标y'''
        ret = [self.obs_prey_before, self.last_obs_x, self.last_obs_y]
        return ret

class Scenario(BaseScenario):
    def __init__(self):
        self.prey_captured = False

    def make_world(self):
        '''创建world,并且添加捕食者(和猎物)到world中 然后初始化他们的位置及观察视野'''
        map_size = FLAGS.map_size
        world = World(width=map_size, height=map_size)

        # list of all agents
        agents = []
        self.atype_to_idx = {
            "predator": [],
            "prey": []
        }
        
        # make agents
        # add predators
        n_predator = FLAGS.n_predator
        for i in range(n_predator):
            agents.append(Predator(i))
            self.atype_to_idx['predator'].append(i)
        
        # add preys
        '''
        n_prey = FLAGS.n_prey
        for i in range(n_prey):
            agents.append(Prey())
            self.atype_to_idx['prey'].append(n_predator + i)
        '''
        
        # used by BaseScenario
        # assign id to agents    捕食者(猎物)都是 agent
        # 把agent添加到world中
        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1
            agent.silent = True # cannot send communication signals

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        '''随机摆放捕食者(和猎物)并初始化观察视野'''
        world.empty_grid()
        # randomly place agents
        for agent in world.agents:
            world.placeObj(agent)
            if agent.itype == 'predator':   # 捕食者还得另外设置猎物观察数据
                agent.reset_obs_prey()
        world.set_observations()
        self.prey_captured = False

    def reward(self, agent, world):
        '''返回一个实数reward'''
        if agent.itype == 'predator':
            if self.prey_captured:
                return 1
            else:
                reward = 0.0
                # 需要看每一个predator是否抓住猎物？如果没抓住reward减小
                for i in self.atype_to_idx['predator']:
                    pred = world.agents[i]
                    check = self.check_prey(pred, world)[0]
                    if check == 0.0:
                        reward -= 0.1
                if reward == 0.0:# 如果所有的predator都围住了猎物，那么相当于抓住了猎物，reward也是1
                    self.prey_captured = True
                    return 1
                return reward
        else: # if prey
            if agent.is_captured(world):     # 为何要减去1？是因为之前多加了一个1？
                return -1
        return 0

    def encode_grid_to_onehot(self, world, grid):
        '''
        state representation plan: one-hot vector per grid cell(with length of number of agents) \n
        id-th index marked when any kind of agent is there \n
        0-th index marked when there is a wall\n
        输出：观察视野中每一个格点的的onehot编码的并联
        '''
        encoded = grid.encode() # full encoded map
        n = len(world.agents) # number of agents

        res = np.array([])
        # encoded是三维array，最后一维有3列，分别表示OBJECT_TO_IDX[v.itype], COLOR_TO_IDX[v.color], v.id
        for cell in encoded.reshape(-1, 3):
            cell_onehot = np.zeros(n + 1)
            if IDX_TO_OBJECT[cell[0]] == 'wall':
                cell_onehot[0] = 1.0
            elif cell[0] != 0:
                cell_onehot[cell[2]] = 1.0
            res = np.concatenate((res, cell_onehot))
        return res  # 观察视野中每一个格点的的onehot编码的并联

    def observation(self, agent, world):
        '''返回world中某个agent的观测情况   \n
        如果agent是捕食者,返回[agent的位置x, y,  0.0/1.0,  True 1.0/False 0.0, prey_x, prey_y]\n
        如果agent是prey, 则返回[prey的位置x, y] \n
        注意, 捕食者的不同感受野反应在了对猎物的观察能力上，没有直接体现在返回obs的数据长度
        '''
        pos_normal = True
        prey_flag = True
        ret = []
        # 获得agent的归一化后的位置
        if pos_normal:
            ret.append(self.get_pos_normal(agent, world))

        if prey_flag:
            # 检测是否看到了猎物
            c_prey = self.check_prey(agent, world)

            if agent.itype == 'predator':
                if c_prey[0] == 1:
                    # 如果是捕食者并且看到了猎物，那么设置捕食者对猎物的观测值
                    agent.set_obs_prey(c_prey[1], c_prey[2])
                # 将对猎物的观测情况添加到状态返回，如果没有观察到猎物，那么猎物观测值应该是旧的数据
                ret.append([c_prey[0]]) # 最近一次有没有观测到，0/1，坐标x，坐标y
                ret.append(agent.get_obs_prey())    # 将最近一次的猎物观测结果1/0, p_x, p_y返回

        ret = np.concatenate(ret)

        return ret

    def get_pos_normal(self, agent, world):
        '''获取某agent的归一化的位置'''
        x, y = agent.pos  # TODO: order has problem
        ret = np.array([x / (world.grid.width-1), y / (world.grid.height-1)])
        return ret

    def check_prey(self, agent, world):
        '''检查是否看到了prey, 返回0.0/1.0, px, py'''
        obs_native = self.encode_grid_to_onehot(world, agent.get_obs())
        check_prey = 0.0
        coor_prey = 0
        cnt = 0
        obs_prey = []
        px = -1.0
        py = -1.0
        for cell in obs_native.reshape(-1, len(world.agents) + 1):
            # one-hot encoded cell w.r.t. agent id
            check = 0.0
            if np.max(cell) != 0:
                idx = np.argmax(cell)
                if idx in [world.agents[i].id for i in self.atype_to_idx['prey']]:
                    check_prey = 1.0
                    check = 1.0
                    coor_prey = cnt
            obs_prey.append(check)
            cnt += 1
        if check_prey == 1.0:
            obs_size = (2*agent.obs_range + 1)
            px = (coor_prey // obs_size) / (obs_size - 1)
            py = (coor_prey % obs_size) / (obs_size - 1)

            if FLAGS.hetero == 2:
                if agent.id in [4]:
                    if not px == 0.5:
                        # print(agent.id, px, py)
                        check_prey = 0.0
                        px = -1.0
                        py = -1.0
        return check_prey, px, py

    def info(self, agent, world):   # 返回全局的信息
        coord_as_state = True
        if coord_as_state:
            # encode coordinates into state
            width = world.grid.width
            height = world.grid.height
            encoded = world.grid.encode()[:, :, 2]
            '''pp环境中，4捕食者1猎物，encoded是全局中agent的位置，7行7列中哪一个格点有agent，那里的值就是该agent的标号
            list of array：
                0:array([0, 0, 0, 0, 2, 4, 0], dtype=int8)
                1:array([5, 0, 0, 0, 0, 0, 0], dtype=int8)
                2:array([0, 0, 0, 0, 0, 0, 0], dtype=int8)
                3:array([0, 3, 0, 0, 0, 1, 0], dtype=int8)
                4:array([0, 0, 0, 0, 0, 0, 0], dtype=int8)
                5:array([0, 0, 0, 0, 0, 0, 0], dtype=int8)
                6:array([0, 0, 0, 0, 0, 0, 0], dtype=int8)
            '''
            state = np.zeros((len(world.agents), 2)) # n_agents * (x,y)
            for x, y in itertools.product(*map(range, (width, height))):
                if encoded[y, x] != 0:
                    state[encoded[y, x] - 1] = np.array([x/width, y/height])
            # 将state 5*2拉平为 1*10，返回这个字典，本质上返回的就是每个agent的pos，那我们可以用ic3net中的车的位置直接代替
            return {'state': state.flatten()}
        else:
            return {'state': self.encode_grid_to_onehot(world, world.grid)}

    def done(self, agent, world):
        '''是否捕获到prey'''
        return self.prey_captured