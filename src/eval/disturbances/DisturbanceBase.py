from walker.multiwalker.multiwalker_base import MultiWalkerEnv


class DisturbanceBase:
    """
    所有扰动类的基类。
    存储了环境和扰动变量。
    """

    def __init__(self, env: MultiWalkerEnv, disturbance_args: dict):
        self.env = env
        self.disturbance_args = disturbance_args

    def start(self):
        pass

    def end(self):
        pass
