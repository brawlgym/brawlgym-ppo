import random

import numpy as np

from brawlgym.utils.action_parsers import DefaultAction, LookupAction
from brawlgym.utils.common_values import BOT_OFF
from brawlgym.utils.reward_functions import CombinedReward

from .reporting import boxed_table

DISCRETE = 0
MULTI_DISCRETE = 1
CONTINUOUS = 2


class BrawlgymWrapper(object):
    """
    Adapts a brawlgym Match to the interface the batched agents drive: array observations,
    a 5-tuple step, and a description of the action space taken from the Match's action parser.

    Fighters can also be handed to pretrained opponents for an episode. Those fighters are drawn
    fresh at every reset, and are dropped from the observations and rewards handed back, so the
    learner never sees them as agents and never trains on their behaviour. `keep_learning` is a
    floor on how many fighters stay with the learning policy, so an episode cannot come out with
    nothing to learn from.

    :param pretrained_agents: {agent: probability}, the chance each individual fighter is given to
                              that agent. Probabilities are drawn per fighter and cumulative, so
                              they should sum to less than 1.
    :param show_lineup: print who is holding each fighter whenever that changes between episodes.
    """

    def __init__(self, match, connect_timeout=300.0, pretrained_agents=None,
                 keep_learning=1, rng=None, show_lineup=True):
        self.match = match
        self.match.connect(connect_timeout)
        self.obs_shape = None
        self.pretrained_agents = dict(pretrained_agents or {})
        self.keep_learning = int(keep_learning)
        self.rng = rng or random.Random()
        self.show_lineup = bool(show_lineup)
        self.occupants = []          # per fighter: None for the learning policy, else the agent
        self.learning = []           # fighter indices the learner still owns
        self._shown = None           # the lineup last printed, so an unchanged one stays quiet
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

        # what to send for a fighter the game itself is driving; the hook ignores it, but the
        # action parser still has to accept it
        if self.action_space_type == DISCRETE:
            table = list(getattr(parser, "table", []))
            self.idle_action = table.index(0) if 0 in table else 0
        else:
            self.idle_action = np.zeros(self.n_actions, dtype=np.float32)

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
        Weighted per-component reward for this step, averaged over the learning agents.
        """
        fn = self.match.reward_function
        if not isinstance(fn, CombinedReward):
            return np.asarray([float(np.mean(rews))], dtype=np.float32)
        weights = np.asarray(fn.reward_weights, dtype=np.float32)
        players = self.match._state.players
        per_agent = [weights * np.asarray(fn.last_rewards[players[i].port], dtype=np.float32)
                     for i in self.learning]
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

    def _draw_occupants(self, n):
        """
        Pick who holds each fighter this episode, leaving at least keep_learning with the learner.
        """
        occupants = [None] * n
        if self.pretrained_agents:
            room = n - self.keep_learning
            for i in self.rng.sample(range(n), n):
                if room <= 0:
                    break
                roll = self.rng.random()
                for agent, chance in self.pretrained_agents.items():
                    roll -= float(chance)
                    if roll < 0:
                        occupants[i] = agent
                        room -= 1
                        break
        return occupants

    def _hand_out_fighters(self, n):
        self.occupants = self._draw_occupants(n)
        self.learning = [i for i, occ in enumerate(self.occupants) if occ is None]
        self.match.set_bot(-1, BOT_OFF)
        for i, occ in enumerate(self.occupants):
            if occ is not None:
                occ.take_over(self.match, i)
        if self.show_lineup:
            self._print_lineup()

    def _print_lineup(self):
        """
        Who is holding each fighter this episode. Printed only when it differs from the last one,
        so a run full of workers stays readable and every line means something changed.
        """
        names = tuple("Learner" if occ is None else repr(occ) for occ in self.occupants)
        if names == self._shown:
            return
        self._shown = names
        players = self.match._state.players
        rows = [[i, names[i], players[i].legend or "?", players[i].team]
                for i in range(len(names))]
        print("\nROLLOUT on port %d\n%s"
              % (self.match.port, boxed_table(["Fighter", "Controller", "Legend", "Team"], rows)),
              flush=True)

    def reset(self):
        obs = np.asarray(self.match.reset(), dtype=np.float32)
        self._hand_out_fighters(len(obs))
        obs = obs[self.learning]
        self.obs_shape = obs.shape
        return obs

    def _full_actions(self, actions):
        """
        Widen the learner's actions back out to one per fighter, asking each pretrained opponent
        for its own. An opponent driven inside the game answers None and gets the idle action.

        Every slot keeps the shape the learner sends, so a scalar an agent returns is broadcast
        into it rather than left as a ragged row.
        """
        if not self.occupants or len(self.learning) == len(self.occupants):
            return actions
        actions = np.asarray(actions)
        full = np.zeros((len(self.occupants),) + actions.shape[1:], dtype=actions.dtype)
        for slot, i in enumerate(self.learning):
            full[i] = actions[slot]
        state = self.match._state
        for i, occ in enumerate(self.occupants):
            if occ is None:
                continue
            chosen = occ.act(state, i)
            full[i] = self.idle_action if chosen is None else chosen
        return full

    def step(self, actions):
        if self.action_space_type != CONTINUOUS:
            actions = np.asarray(actions).astype(np.int32)
        obs, rews, done, state = self.match.step(self._full_actions(actions))
        obs = np.asarray(obs, dtype=np.float32)[self.learning]
        rews = [float(rews[i]) for i in self.learning]
        info = {"state": state, "reward_components": self.reward_components(rews),
                "action_frequencies": self.action_frequencies(actions)}
        return obs, rews, done, False, info

    def close(self):
        self.match.close()
