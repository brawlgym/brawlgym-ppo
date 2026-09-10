import numpy as np
from brawlgym.utils.gamestates import GameState
from brawlgym_ppo.util import MetricsLogger


class ExampleLogger(MetricsLogger):
    def _collect_metrics(self, game_state: GameState) -> list:
        return [np.asarray([p.damage for p in game_state.players]),
                np.asarray([p.dead for p in game_state.players], dtype=np.float32)]

    def _report_metrics(self, collected_metrics, wandb_run, cumulative_timesteps):
        avg_damage = 0
        avg_dead = 0
        for metric_array in collected_metrics:
            avg_damage += np.mean(metric_array[0])
            avg_dead += np.mean(metric_array[1])
        avg_damage /= len(collected_metrics)
        avg_dead /= len(collected_metrics)
        report = {"avg_damage": avg_damage,
                  "avg_dead": avg_dead,
                  "Cumulative Timesteps": cumulative_timesteps}
        wandb_run.log(report)


def build_brawlgym_env(port):
    import brawlgym
    from brawlgym.utils.reward_functions import CombinedReward, ComboReward, DamageDealtReward, DamageTakenPenalty, GroundedPenalty, KOReward, VelocityReward, WhiffPenalty
    from brawlgym.utils.obs_builders import DefaultObs
    from brawlgym.utils.terminal_conditions import TeamWipeCondition, TimeoutCondition
    from brawlgym.utils.action_parsers import LookupAction
    from brawlgym.utils.state_setters import RandomStateSetter, ArmedStateSetter

    n_players = 2
    game_fps = 60
    tick_skip = 4
    timeout_seconds = 60
    timeout_steps = int(round(timeout_seconds * game_fps / tick_skip))

    action_parser = LookupAction()
    terminal_conditions = [TeamWipeCondition(), TimeoutCondition(timeout_steps)]
    rewards_to_combine = (DamageDealtReward(),
                          DamageTakenPenalty(),
                          KOReward(ko_reward=100.0, death_penalty=100.0),
                          WhiffPenalty(penalty=3.0),
                          ComboReward(link_reward=5.0, max_gap=15),
                          VelocityReward(),
                          GroundedPenalty())
    reward_weights = (1.0, 1.0, 1.0, 1.0, 1.0, 0.005, 0.00025)

    reward_fn = CombinedReward(rewards_to_combine, reward_weights)
    obs_builder = DefaultObs()
    state_setter = ArmedStateSetter(RandomStateSetter(), arm_chance=0.75)

    env = brawlgym.make(tick_skip=tick_skip,
                        n_players=n_players,
                        legends=("Mordex", "Nix"),
                        terminal_conditions=terminal_conditions,
                        reward_function=reward_fn,
                        obs_builder=obs_builder,
                        action_parser=action_parser,
                        state_setter=state_setter,
                        port=port,
                        auto_mute=True,
                        auto_minimize=True,
                        game_speed=0)

    return env


if __name__ == "__main__":
    from brawlgym_ppo import Learner
    metrics_logger = ExampleLogger()

    # one game instance per process
    n_proc = 8

    # educated guess - could be slightly higher or lower
    min_inference_size = max(1, int(round(n_proc * 0.9)))

    learner = Learner(build_brawlgym_env,
                      n_proc=n_proc,
                      n_players=2,
                      minimize_game_windows=True,
                      mute_game_audio=True,
                      min_inference_size=min_inference_size,
                      metrics_logger=metrics_logger,
                      policy_layer_sizes=(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024),
                      critic_layer_sizes=(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024),
                      policy_lr=2e-4,
                      critic_lr=2e-4,
                      ppo_batch_size=50000,
                      ts_per_iteration=50000,
                      exp_buffer_size=150000,
                      ppo_minibatch_size=50000,
                      ppo_ent_coef=0.005,
                      ppo_epochs=3,
                      standardize_returns=True,
                      standardize_obs=False,
                      save_every_ts=5_000_000,
                      timestep_limit=1_000_000_000,
                      log_to_wandb=True)
    learner.learn()
