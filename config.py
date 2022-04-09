#!/usr/bin/env python
# coding=utf8
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import tensorflow as tf
import logging
import time
import envs.config_env as config_env
import agents.config_agents as config_agent

flags = tf.flags
# SCHDEDNET参数
flags.DEFINE_integer("seed", 0, "Random seed number")
flags.DEFINE_string("folder", "default", "Result file folder name")
flags.DEFINE_bool("display", False, "whether to display")
# IC3NET参数
flags.DEFINE_string('nactions', '1', 
                    'the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
flags.DEFINE_float('action_scale', 1.0, 'scale action output from model')
flags.DEFINE_string('save', '', 'save the model after training')
flags.DEFINE_integer('save_every', 0, 'save the model after every n_th epoch')
flags.DEFINE_string('load', '', 'load the model')
flags.DEFINE_bool('display_curses', False, 'Display environment state')
flags.DEFINE_integer('nagents', 1, "Number of agents (used in multiagent)")
flags.DEFINE_bool('comm_action_one', False, 'Whether to always talk, sanity check for hard attention.')
flags.DEFINE_integer('detach_gap', 10000,
                     'detach hidden state and cell state for rnns at this interval.'
                    + ' Default 10000 (very high)')
flags.DEFINE_bool('share_weights',False, 'Share weights for hops')
# IC3NET中通过 .运算 自动添加的参数，这里手动添加
flags.DEFINE_integer('num_actions', 0, 'num_actions')
flags.DEFINE_integer('dim_actions', 0, 'dim_actions')
flags.DEFINE_integer('num_inputs', 0, 'num_inputs')
# parser.add_argument('--advantages_per_action', default=False, action='store_true',
#                     help='Whether to multipy log porb for each chosen action with advantages')

# 配置环境和智能体!!!
config_env.config_env(flags)    # 或许可以不用？
config_agent.config_agent(flags)

# Make < result file > with given filename
now = time.localtime()
s_time = "%02d%02d%02d%02d%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
file_name = str(flags.FLAGS.n_predator) + "-"
file_name += config_env.get_filename() + "-" + config_agent.get_filename()
file_name += "-seed-"+str(flags.FLAGS.seed)+"-" + s_time

# result是一个Logger
result = logging.getLogger('Result')
result.setLevel(logging.INFO)

if flags.FLAGS.folder == "default":
    log_filename = "./results/eval/r-" + file_name + ".txt"
    # FileHandler: Open the specified file and use it as the stream for logging.
    result_fh = logging.FileHandler("./results/eval/r-" + file_name + ".txt")
    nn_filename = "./results/nn/n-" + file_name
else:
    log_filename = "./results/eval/"+ flags.FLAGS.folder +"/r-" + file_name + ".txt"
    result_fh = logging.FileHandler("./results/eval/"+ flags.FLAGS.folder +"/r-" + file_name + ".txt")
    nn_filename = "./results/nn/" + flags.FLAGS.folder + "/n-" + file_name
# 一个result_fh的例子：
'''
r
-4-
s-predator_prey_obs-map-7-or-1
-a-schednet-clr-0.0001-alr-1e-05-hc-64-cp-2-ss-False-es-False-as-False-sn-1-sched-schedule-s_type-top
-seed-0-0324161005.txt
'''
# 把文件header加到result这个logger中，以后的相关信息便会log到这个文件中
result_fm = logging.Formatter('[%(filename)s:%(lineno)s] %(asctime)s\t%(message)s')
result_fh.setFormatter(result_fm) # Set the formatter for this handler.
result.addHandler(result_fh)


# Used to map colors to integers
COLOR_TO_IDX = {
    'red'   : 0,
    'green' : 1,
    'blue'  : 2,
    'purple': 3,
    'yellow': 4,
    'grey'  : 5
}

IDX_TO_COLOR = dict(zip(COLOR_TO_IDX.values(), COLOR_TO_IDX.keys()))

# Map of object type to integers
OBJECT_TO_IDX = {
    'empty'         : 0,
    'wall'          : 1,
    'agent'         : 2,
    'predator'      : 3,
    'prey'          : 4
}

IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))