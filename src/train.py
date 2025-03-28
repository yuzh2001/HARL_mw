"""Train an algorithm."""

import rich.pretty
import wandb
import hydra
from omegaconf import DictConfig
import omegaconf
import rich
from harl.runners import RUNNER_REGISTRY


def _to_dict(cfg1: DictConfig):
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rich.pretty.pprint(cfg, expand_all=True)

    # 1. 从配置里读取参数，转换为harl使用的格式
    algo_args = cfg.algorithm
    env_args = cfg.environment

    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.environment.scenario

    # 2. 使用当前时间生成名称
    from datetime import datetime

    ts = datetime.now().strftime("%m%d-%H%M")
    run_name = f"[{algorithm_name}]<{scenario_name}>"

    group_name = cfg.group_name
    if group_name == "latest":
        group_name = datetime.now().strftime("%m%d/%H%M")

    algo_args.logger.log_dir = f"./results/{group_name}"
    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }

    if (
        env_name == "pettingzoo_mw"
        and algo_args.train.get("episode_length") is not None
    ):
        algo_args.train.episode_length = 500

    # 3. 初始化wandb
    wandb.init(
        project="HARL",
        config=_to_dict(cfg),
        sync_tensorboard=True,
        name=run_name + f"_{ts}",
    )

    # 4. 整理参数，转换为dict以传导给harl
    algo_dict = _to_dict(algo_args)
    del algo_dict["name"]

    env_dict = _to_dict(env_args)
    del env_dict["name"]
    del env_dict["scenario"]

    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    # 5. 启动训练
    runner.run()

    # 6. 训练完成
    runner.close()
    import requests

    requests.get(f"https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/{run_name}训练完成")
    wandb.finish()


if __name__ == "__main__":
    main()
