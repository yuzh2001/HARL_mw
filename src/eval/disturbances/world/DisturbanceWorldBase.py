from disturbances import DisturbanceBase, MultiWalkerEnv


class DisturbanceWorldBase(DisturbanceBase):
    """
    对世界做扰动的基类。

    提供colorize和un_colorize方法，在被扰动时，包裹会变色。
    """

    def __init__(self, env: MultiWalkerEnv, disturbance_args: dict):
        super().__init__(env, disturbance_args)
        self.color1 = (255, 0, 0)
        self.color2 = (255, 0, 0)

    def colorize(self, color1: tuple = (255, 0, 0), color2: tuple = (255, 0, 0)):
        for ter in self.env.terrain:
            ter.color1 = color1
            ter.color2 = color2

    def un_colorize(self):
        for index, ter in enumerate(self.env.terrain):
            color = (76, 255 if index % 2 == 0 else 204, 76)
            ter.color1 = color
            ter.color2 = color

    def start(self):
        self.colorize()
        pass

    def end(self):
        self.un_colorize()
        pass
