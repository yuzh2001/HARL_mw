from disturbances import DisturbanceBase, MultiWalkerEnv
from disturbances.package import DisturbancePackageWeightChange
from disturbances.walker import DisturbanceWalkerMotorSpeedChange
from disturbances.world import DisturbanceWorldFrictionChange

disturbance_dict = {
    "weight_change": DisturbancePackageWeightChange,
    "world_friction_change": DisturbanceWorldFrictionChange,
    "walker_motor_speed_change": DisturbanceWalkerMotorSpeedChange,
}


class DisturbanceFactory:
    """
    承载了两个任务：
    1. 从json向类的具体转换；
    2. 记载执行时机并执行。
    """

    def __init__(
        self,
        base_env: MultiWalkerEnv,
        name: str,
        start_at: int,
        end_at: int,
        disturbance_args: dict,
    ):
        self.env = base_env
        self.disturbance: DisturbanceBase = (
            DisturbanceFactory._get_disturbance_func_from_dict(name)(
                env=self.env,
                disturbance_args=disturbance_args,
            )
        )
        self.start_at = start_at
        self.end_at = end_at
        self.disturbance_args = disturbance_args

    def _get_disturbance_func_from_dict(disturbance_name: str) -> DisturbanceBase:
        return disturbance_dict[disturbance_name]

    def execute_with_frame(self, frame: int):
        if frame == self.start_at:
            self.start()
        elif frame == self.end_at:
            self.recover()

    def start(self):
        self.disturbance.start()

    def recover(self):
        self.disturbance.end()
