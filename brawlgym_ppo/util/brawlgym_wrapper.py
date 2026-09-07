import numpy as np

from brawlgym.utils.action_parsers import DefaultAction, LookupAction

DISCRETE = 0
MULTI_DISCRETE = 1
CONTINUOUS = 2


class BrawlgymWrapper(object):
    """
    Adapts a brawlgym Match to the interface the batched agents drive: array observations,
    a 5-tuple step, and a description of the action space taken from the Match's action parser.
    """

    def __init__(self, match, connect_timeout=300.0):
        self.match = match
        self.match.connect(connect_timeout)
        self.obs_shape = None

        parser = match.action_parser
        if isinstance(parser, LookupAction):
            self.action_space_type = DISCRETE
            self.n_actions = parser.n_actions
            self.bins = []
        elif isinstance(parser, DefaultAction):
            self.action_space_type = MULTI_DISCRETE
            self.n_actions = parser.get_action_space_size()
            self.bins = [2] * self.n_actions
        else:
            self.action_space_type = CONTINUOUS
            self.n_actions = parser.get_action_space_size()
            self.bins = []

    @property
    def state(self):
        return self.match._state

    def reset(self):
        obs = np.asarray(self.match.reset(), dtype=np.float32)
        self.obs_shape = obs.shape
        return obs

    def step(self, actions):
        if self.action_space_type != CONTINUOUS:
            actions = np.asarray(actions).astype(np.int32)
        obs, rews, done, state = self.match.step(actions)
        obs = np.asarray(obs, dtype=np.float32)
        rews = [float(r) for r in rews]
        info = {"state": state}
        return obs, rews, done, False, info

    def close(self):
        self.match.close()
