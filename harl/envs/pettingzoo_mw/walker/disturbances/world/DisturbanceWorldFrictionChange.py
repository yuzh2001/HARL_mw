from harl.envs.pettingzoo_mw.walker.disturbances.world import DisturbanceWorldBase


class DisturbanceWorldFrictionChange(DisturbanceWorldBase):
    """
    对世界摩擦力做扰动的类。

    disturbance_args: dict = {"friction": 4.57}
    """

    def start(self):
        super().start()
        for ter in self.env.terrain:
            ter.fixtures[0].friction = self.disturbance_args["friction"]

    def end(self):
        DEFAULT_FRICTION = 2.5
        for ter in self.env.terrain:
            ter.fixtures[0].friction = DEFAULT_FRICTION
        super().end()
