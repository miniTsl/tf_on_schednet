#!/usr/bin/env python
# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# 下一步要研究的代码是schednet的world部分怎么实现的，如何与ic3融合。IC3的网络部分做的隔离度好一些，
# 需要改的地方，agent的观测obs_n（改IC3net），predator的动作要改成车的动作,动作域！！！，train的管控流程

# SCHEDNET中的内容
import make_env
import config
import agents
from envs.environment import MultiAgentEnv
import envs.scenarios as scenarios

# IC3NET中的内容
from ic3net_envs import traffic_junction_env
from env_wrappers import *
from utils import *

import sys
import gym
import logging
import time
import random
import tensorflow as tf
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    return None

# 配置并用FLAGS保存命令行参数
FLAGS = config.flags.FLAGS 
'''
a_lr:1e-05
a_share:False
agent:'schednet'
alsologtostderr:False
b_size:10000
c_lr:0.0001
capa:2
comm:5
df:0.9
e_share:False
eval_on_train:True
eval_step:2500
folder:'default'
gui:False
h_critic:64
hetero:1
history_len:1
load_nn:False
log_dir:''
logger_levels:{}
logtostderr:False
m_size:64
map_size:7
max_step:500
moving_prey:True
n_predator:4
n_prey:1
nn_file:''
obs_diagonal:True
obs_range:1
only_check_args:False
op_conversion_fallback_to_while_loop:False
pdb:False
pdb_post_mortem:False
pre_train_step:10
profile_file:None
render_every:1000.0
run_with_pdb:False
run_with_profiling:False
s_num:1
s_share:False
scenario:'predator_prey_obs'
sch_type:'top'
sched:'schedule'
seed:0
showprefixforinfo:True
stderrthreshold:'fatal'
tau:0.05
test_random_seed:301
test_randomize_ordering_seed:''
test_srcdir:''
test_tmpdir:'C:\\Users\\Tesla\\AppData\\Local\\Temp\\absl_testing'
testing_step:2500
train:True
trainable_encoder:True
training_step:800000
use_action_in_critic:False
use_cprofile_for_profiling:True
v:-1
verbosity:-1
w_lr:1e-05
xml_output_file:''
'''

# 设置随机种子
set_seed(1)
FLAGS.comm_action_one = True
'''
可能有用：
FLAGS.nfriendly = FLAGS.nagents
if hasattr(FLAGS, 'enemy_comm') and FLAGS.enemy_comm:
    if hasattr(FLAGS, 'nenemies'):
        FLAGS.nagents += FLAGS.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")
'''

# 创建环境
# 原来env = make_env.make_env(FLAGS.scenario)   #scenario:'predator_prey_obs'，现在改用IC3net的创建方法
scenario = scenarios.load(FLAGS.scenario + ".py").Scenario()
# create world：scenario的方法
world = scenario.make_world()
env = traffic_junction_env.TrafficJunctionEnv(world, observation_callback=scenario.observation, info_callback = scenario.info)    
env.multi_agent_init(FLAGS)  # 初始化多智能体参数
env = GymWrapper(env) 
# if FLAGS.display:    # 无关紧要
#     env.init_curses()

# 下面用到了wrapper中的三个静态方法
FLAGS.num_inputs = env.observation_dim    # for multi-agent, this is the obs per agent,ic3net中是61（medium）
FLAGS.dim_actions = env.dim_actions     # for multi-agent, 这是每个agent的action空间的维度，一般只有一维，就是action维度，commnet中通信维度默认是0，如果加上的话dim_actions=2
FLAGS.num_actions = env.num_actions     # 每个aent的动作数量，对于ic3net中车来说来说，只有两种：前进、停止
# Multi-action
if not isinstance(FLAGS.num_actions, (list, tuple)): # single action case
    FLAGS.num_actions = [FLAGS.num_actions] # list化

# 获得两个不同名字的logger：环境 logger和智能体 logger
logger_env = logging.getLogger('GridMARL')
logger_agent = logging.getLogger('Agent')
# log命令行中的一些信息到Logger中。
# (method) info: (msg: object, *args: object)……
# info: Log 'msg % args' with severity 'INFO'.
logger_agent.info('Agent: {}'.format(FLAGS.agent))


# 创建训练器,加载agents文件夹下FLAGS.agent模型文件夹中的trainer.py文件
# 注意env此时是一个env_wrapper对象,成员env才是schednet中的env
trainer = agents.load(FLAGS.agent + "/trainer.py").Trainer(env)
# 打印(schednet, 4-s-predator_prey_obs-map-7-or-1-a-schednet-clr-0.0001-alr-1e-05-hc-64-cp-2-ss-False-es-False-as-False-sn-1-sched-schedule-s_type-top-seed-0-……)
print(FLAGS.agent, config.file_name)

# （开始训练，）然后test
if FLAGS.train:
    start_time = time.time()
    trainer.learn()
    finish_time = time.time()
    trainer.test()
    print("TRAINING TIME (sec)", finish_time - start_time)
else:
    trainer.test()
    
# 打印保存的日志文件的名字，便于查看result
print("LOG_FILE:\t" + config.log_filename)