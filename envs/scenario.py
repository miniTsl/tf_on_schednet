import numpy as np

# defines scenario upon which the world is built
class BaseScenario(object):
    # create elements of the world
    def make_world(self):
        # Method or function hasn't been implemented yet.
        raise NotImplementedError()
    # create initial conditions of the world
    def reset_world(self, world):
        raise NotImplementedError()
    def reward(self, agent, world):
        raise NotImplementedError()
    def observation(self, agent, world):
        raise NotImplementedError()
    def info(self, agent, world):
        raise NotImplementedError()
    def done(self, agent, world):
        raise NotImplementedError()