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


class Prey(CoreAgent):
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

class Predator(CoreAgent):
    def __init__(self):
        super(Predator, self).__init__('predator', 'blue')
        self.obs_range = 1

class Scenario(BaseScenario):
    def __init__(self):
        self.prey_captured = False

    def make_world(self):
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
            agents.append(Predator())
            self.atype_to_idx['predator'].append(i)

        # add preys
        n_prey = FLAGS.n_prey
        for i in range(n_prey):
            agents.append(Prey())
            self.atype_to_idx['prey'].append(n_predator + i)

        # used by BaseScenario
        # assign id to agents
        world.agents = agents
        for i, agent in enumerate(world.agents):
            agent.id = i + 1
            agent.silent = True

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.empty_grid()

        # randomly place agents
        for agent in world.agents:
            world.placeObj(agent)
        
        world.set_observations()

        self.prey_captured = False

    def reward(self, agent, world):
        if agent.itype == 'predator':
            if self.prey_captured:
                return 1
            else:
                reward = -0.01
                # determine whether the prey has been captured
                for i in self.atype_to_idx['prey']:
                    prey = world.agents[i]
                    if prey.is_captured(world):
                        self.prey_captured = True
                        return 1
                return reward
        else: # if prey
            if agent.is_captured(world):
                return -1
        return 0

    def encode_grid_to_onehot(self, world, grid):
        encoded = grid.encode() # full encoded map
        # state representation plan: one-hot vector per grid cell
        # id-th index marked when any kind of agent is there
        # 0-th index marked when there is a wall
        n = len(world.agents) # number of agents

        res = np.array([])
        for cell in encoded.reshape(-1, 3):
            cell_onehot = np.zeros(n + 1)
            if IDX_TO_OBJECT[cell[0]] == 'wall':
                cell_onehot[0] = 1.0
            elif cell[0] != 0:
                cell_onehot[cell[2]] = 1.0
            res = np.concatenate((res, cell_onehot))

        return res

    def observation(self, agent, world):
        # obs_native = np.array(agent.get_obs())
        obs_native = self.encode_grid_to_onehot(world, agent.get_obs())
        # encode all predators and preys into same id
        # TODO: try not to distinguish the same kind of agents..
        indistinguish = True
        if indistinguish:
            obs = np.array([])
            for cell in obs_native.reshape(-1, len(world.agents) + 1):
                # one-hot encoded cell w.r.t. agent id
                compact_cell = np.zeros(3) # wall, predator, prey
                if np.max(cell) != 0:
                    idx = np.argmax(cell)
                    if idx == 0: # wall
                        compact_cell[0] = 1.0
                    elif idx in [world.agents[i].id for i in self.atype_to_idx['predator']]:
                        compact_cell[1] = 1.0
                    elif idx in [world.agents[i].id for i in self.atype_to_idx['prey']]:
                        compact_cell[2] = 1.0
                    else:
                        raise Exception('cell has to be wall/predator/prey!')
                obs = np.concatenate([obs, compact_cell])
            ret = obs
        else:
            ret = obs_native
        if not FLAGS.obs_diagonal:
            blk = ret.reshape([-1, 3])
            ret = np.concatenate([[blk[1]], blk[3:6], [blk[7]]]).flatten()
        # encode current position into observation
        x, y = agent.pos
        ret = np.concatenate([ret, [x / world.grid.width, y / world.grid.height]])
        return ret

    def info(self, agent, world):
        # info() returns the global state
        coord_as_state = True
        if coord_as_state:
            # encode coordinates into state
            width = world.grid.width
            height = world.grid.height
            encoded = world.grid.encode()[:, :, 2]
            state = np.zeros((len(world.agents), 2)) # n_agents * (x,y)
            for x, y in itertools.product(*map(range, (width, height))):
                if encoded[y, x] != 0:
                    state[encoded[y, x] - 1] = np.array([x/width, y/height])
            return {'state': state.flatten()}
        else:
            return {'state': self.encode_grid_to_onehot(world, world.grid)}

    def done(self, agent, world):
        return self.prey_captured