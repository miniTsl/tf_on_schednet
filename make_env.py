"""
Code for creating a multiagent environment with one of the scenarios listed
in ./scenarios/.
Can be called by using, for example:
    env = make_env('simple_speaker_listener')
After producing the env object, can be used similarly to an OpenAI gym
environment.

A policy using this environment must output actions in the form of a list
for all agents. Each element of the list should be a numpy array,
of size (env.world.dim_p + env.world.dim_c, 1). Physical actions precede
communication actions in this array. See environment.py for more details.
"""

def make_env(scenario_name):
    '''
    产生的环境是一个openAI环境env（一个类对象），可以调用env.reset() 和env.step().进行交互；
    策略输入环境的是一个list，list中每个元素是某个agent的action，物理action优先，通信action在后；
    返回的是：.observation_space（每个agent的观测空间），.action_space每个agent的action空间，.nagent的数量；
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from envs.environment import MultiAgentEnv
    import envs.scenarios as scenarios

    # load scenario from script
    # scenario是一个类对象,已经实现了env.ENV中的部分需要重载的函数
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world：scenario方法
    world = scenario.make_world()
    # create multiagent environment
    # 需要提供world环境以及相应的方法函数这样就可以搭建env
    env = MultiAgentEnv(world, reset_callback=scenario.reset_world, 
                                reward_callback=scenario.reward, 
                                observation_callback=scenario.observation,
                                info_callback=scenario.info,
                                done_callback=scenario.done,)
    return env
