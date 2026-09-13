"""
A saved PPO checkpoint, replayed as an opponent.
"""
import os

import numpy as np
import torch

from brawlgym.utils.action_parsers import LookupAction, mask_to_buttons
from brawlgym.utils.agents import HardcodedAgent
from brawlgym.utils.common_values import NUM_BUTTONS
from brawlgym.utils.obs_builders import DefaultObs

from ..ppo.discrete_policy import DiscreteFF


class PolicyAgent(HardcodedAgent):
    """
    Acts from a PPO_POLICY.pt saved by the learner.

    The network shape is read off the checkpoint, so only the folder is needed. It carries its own
    observation builder and action table: a checkpoint is fixed at whatever it was trained against
    while the learning policy's observation keeps moving, so pass the pair that matches the run
    this checkpoint came from. The observation width is checked against the network's input layer
    the first time it acts.

    :param checkpoint: a learner checkpoint folder, or the PPO_POLICY.pt inside one.
    :param obs_builder: the observation this checkpoint expects.
    :param action_parser: the action table this checkpoint's outputs index into.
    :param deterministic: take the most likely action instead of sampling.
    """

    def __init__(self, checkpoint, obs_builder=None, action_parser=None,
                 deterministic=False, device="cpu"):
        path = checkpoint
        if os.path.isdir(path):
            path = os.path.join(path, "PPO_POLICY.pt")
        self.path = path
        self.device = device
        self.deterministic = bool(deterministic)
        self.obs_builder = obs_builder if obs_builder is not None else DefaultObs()
        self.action_parser = action_parser if action_parser is not None else LookupAction()

        state = torch.load(path, map_location=device)
        weights = [v for k, v in state.items() if k.endswith(".weight")]
        self.obs_size = int(weights[0].shape[1])
        self.n_actions = int(weights[-1].shape[0])

        self.policy = DiscreteFF(self.obs_size, self.n_actions,
                                 [int(w.shape[0]) for w in weights[:-1]], device)
        self.policy.load_state_dict(state)
        self.policy.eval()

        self._prev = {}

    def take_over(self, match, fighter):
        """
        Point the observation builder at this match and clear the action history it keeps.
        """
        if hasattr(self.obs_builder, "set_map_info"):
            self.obs_builder.set_map_info(match.map_info)
        self.obs_builder.reset(match._state)
        self._prev = {}

    def act(self, state, player_index):
        player = state.players[player_index]
        prev = self._prev.get(player.port)
        if prev is None:
            prev = np.zeros(NUM_BUTTONS, dtype=np.float32)

        obs = np.asarray(self.obs_builder.build_obs(player, state, prev), dtype=np.float32)
        if obs.shape[-1] != self.obs_size:
            raise ValueError("%s takes %d observation features, the builder produced %d"
                             % (os.path.basename(self.path), self.obs_size, obs.shape[-1]))

        with torch.no_grad():
            action, _ = self.policy.get_action(obs, deterministic=self.deterministic)
        action = int(np.asarray(action).reshape(-1)[0])

        mask = self.action_parser.parse_actions([action], state)[0]
        self._prev[player.port] = np.asarray(mask_to_buttons(mask), dtype=np.float32)
        return action

    def __repr__(self):
        return "PolicyAgent(%s)" % os.path.basename(os.path.dirname(self.path))
