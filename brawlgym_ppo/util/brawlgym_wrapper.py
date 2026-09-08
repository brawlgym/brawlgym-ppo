import numpy as np

from brawlgym.utils.action_parsers import DefaultAction, LookupAction
from brawlgym.utils.reward_functions import CombinedReward

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
        self.reward_names = self._reward_names(match.reward_function)
        self.action_names = []

        parser = match.action_parser
        if isinstance(parser, LookupAction):
            self.action_space_type = DISCRETE
            self.n_actions = parser.n_actions
            self.bins = []
            self.action_names = list(parser.action_names)
        elif isinstance(parser, DefaultAction):
            self.action_space_type = MULTI_DISCRETE
            self.n_actions = parser.get_action_space_size()
            self.bins = [2] * self.n_actions
        else:
            self.action_space_type = CONTINUOUS
            self.n_actions = parser.get_action_space_size()
            self.bins = []

    @staticmethod
    def _reward_names(fn):
        """
        One name per reward component: the class names of a CombinedReward's parts (numbered when a
        class appears more than once), or the single reward's class name.
        """
        parts = fn.reward_functions if isinstance(fn, CombinedReward) else (fn,)
        names = [type(f).__name__ for f in parts]
        out = []
        for i, n in enumerate(names):
            out.append(n if names.count(n) == 1 else "%s_%d" % (n, names[:i + 1].count(n)))
        return out

    def reward_components(self, rews):
        """
        Weighted per-component reward for this step, averaged over the agents.
        """
        fn = self.match.reward_function
        if not isinstance(fn, CombinedReward):
            return np.asarray([float(np.mean(rews))], dtype=np.float32)
        weights = np.asarray(fn.reward_weights, dtype=np.float32)
        per_agent = [weights * np.asarray(fn.last_rewards[p.port], dtype=np.float32)
                     for p in self.match._state.players]
        return np.mean(per_agent, axis=0)

    def action_frequencies(self, actions):
        """
        Fraction of agents that picked each discrete action this step (empty for other spaces).
        """
        if not self.action_names:
            return np.zeros(0, dtype=np.float32)
        idx = np.asarray(actions).astype(np.int64).reshape(-1)
        return (np.bincount(idx, minlength=len(self.action_names)) / max(1, idx.size)).astype(np.float32)

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
        info = {"state": state, "reward_components": self.reward_components(rews),
                "action_frequencies": self.action_frequencies(actions)}
        return obs, rews, done, False, info

    def close(self):
        self.match.close()
