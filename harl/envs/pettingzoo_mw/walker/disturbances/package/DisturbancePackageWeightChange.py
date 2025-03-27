from harl.envs.pettingzoo_mw.walker.disturbances.package import DisturbancePackageBase


class DisturbancePackageWeightChange(DisturbancePackageBase):
    """
    对包裹质量做扰动的类。

    disturbance_args: dict = {"mass": 4.57}
    """

    def start(self):
        super().start()
        self.env.package.mass = self.disturbance_args["mass"]

    def end(self):
        DEFAULT_MASS = 4.57
        self.env.package.mass = DEFAULT_MASS
        super().end()
