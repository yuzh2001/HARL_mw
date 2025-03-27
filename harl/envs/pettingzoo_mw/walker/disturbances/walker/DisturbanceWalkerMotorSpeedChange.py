from harl.envs.pettingzoo_mw.walker.disturbances.walker import DisturbanceWalkerBase


class DisturbanceWalkerMotorSpeedChange(DisturbanceWalkerBase):
    """
    对双足机器人电机速度做扰动的类。

    disturbance_args: dict = {"speed_factor_hip": 4, "speed_factor_knee": 6}
    """

    def start(self):
        super().start()
        if self.disturbance_args.get("effect_on_agent") is None:
            for walker in self.env.agents:
                walker.speed_factor_hip = self.disturbance_args["speed_factor_hip"]
                walker.speed_factor_knee = self.disturbance_args["speed_factor_knee"]
        else:
            for agent_id in self.disturbance_args["effect_on_agent"]:
                self.env.agents[agent_id].speed_factor_hip = self.disturbance_args[
                    "speed_factor_hip"
                ]
                self.env.agents[agent_id].speed_factor_knee = self.disturbance_args[
                    "speed_factor_knee"
                ]

    def end(self):
        DEFAULT_SPEED = [4, 6]
        if self.disturbance_args.get("effect_on_agent") is None:
            for walker in self.env.agents:
                walker.speed_factor_hip = DEFAULT_SPEED[0]
                walker.speed_factor_knee = DEFAULT_SPEED[1]
        else:
            for agent_id in self.disturbance_args["effect_on_agent"]:
                self.env.agents[agent_id].speed_factor_hip = DEFAULT_SPEED[0]
                self.env.agents[agent_id].speed_factor_knee = DEFAULT_SPEED[1]
        super().end()
