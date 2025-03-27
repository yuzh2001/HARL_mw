from harl.common.base_logger import BaseLogger
import time


class PettingZooMWLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PettingZooMWLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1

    def get_task_name(self):
        return "mw_pettingzoo"

    def eval_init(self):
        super().eval_init()
