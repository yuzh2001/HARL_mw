from harl.common.base_logger import BaseLogger


class PettingZooMWLogger(BaseLogger):
    def get_task_name(self):
        return "mw_pettingzoo"
