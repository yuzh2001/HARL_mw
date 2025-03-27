from harl.common.base_logger import BaseLogger
from .pettingzoo_mw_env import PettingZooMWEnv
from typing import List


class PettingZooMWLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PettingZooMWLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False
        self.test_data = {"terminate_at": [], "angle_data": [], "package_x": []}

    def get_task_name(self):
        return "mw_pettingzoo"

    def eval_init(self):
        super().eval_init()

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        if not self.test_data:
            super().eval_per_step(eval_data)
        else:
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = eval_data
            self.test_data["angle_data"].extend(
                eval_infos[i][0]["package_angle"] for i in range(len(eval_infos))
            )
            for i in range(len(eval_infos)):
                if eval_dones[i][0]:
                    self.test_data["terminate_at"].append(eval_infos[i][0]["curr_step"])
                    self.test_data["package_x"].append(eval_infos[i][0]["package_x"])
            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.eval_infos = eval_infos
