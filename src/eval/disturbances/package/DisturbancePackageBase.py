from disturbances import DisturbanceBase, MultiWalkerEnv


class DisturbancePackageBase(DisturbanceBase):
    """
    对包裹做扰动的基类。

    提供colorize和un_colorize方法，在被扰动时，包裹会变色。
    """

    def __init__(self, env: MultiWalkerEnv, disturbance_args: dict):
        super().__init__(env, disturbance_args)
        self.color1 = (255, 0, 0)
        self.color2 = (255, 0, 0)

    def colorize(self, color1: tuple = (255, 0, 0), color2: tuple = (255, 0, 0)):
        self.env.package.color1 = color1
        self.env.package.color2 = color2

    def un_colorize(self):
        self.env.package.color1 = (127, 102, 229)
        self.env.package.color2 = (76, 76, 127)

    def start(self):
        self.colorize()
        pass

    def end(self):
        self.un_colorize()
        pass
