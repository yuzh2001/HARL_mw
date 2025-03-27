from __future__ import annotations

import json
import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rich
from matplotlib import font_manager
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import omegaconf
from copy import deepcopy
import wandb
# from .disturbances import DisturbanceFactory, MultiWalkerEnv
# from .utils.gif import export_gif
# from walker import multiwalker_v9

from harl.runners import RUNNER_REGISTRY
from harl.envs.pettingzoo_mw.pettingzoo_mw_logger import PettingZooMWLogger
import requests
import hydra_type
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra

os.environ["SDL_VIDEODRIVER"] = "dummy"

# 设置字体路径
# font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 确保路径正确
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
wandb_results = []
max_cycles = 500


def _to_dict(cfg1: DictConfig):
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)


def log_wandb():
    angle_intervals = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 5),
        (5, 8),
        (8, 10),
        (10, 15),
        (15, float("inf")),
    ]
    columns = ["algo", "variant", "scenario", "terminate_cnt"]
    columns.extend([f"angle-{start}-{end}" for start, end in angle_intervals])
    # 将字典数据转换为列表格式
    table_data = []
    for result in wandb_results:
        table_data.append(
            [
                result["algo"],
                result["variant"],
                result["scenario"],
                result["terminate_cnt"],
                *[
                    result["angle_data"].get(f"angle-{start}-{end}", 0)
                    for start, end in angle_intervals
                ],
            ]
        )

    test_table = wandb.Table(data=table_data, columns=columns)
    wandb.log({"test_table": test_table})


def run_evaluations(
    config: hydra_type.EvalConfig, checkpoint: hydra_type.CheckpointConfig
) -> tuple[dict, dict]:
    """执行baseline和扰动测试的评估"""
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
    )
    os.makedirs(gif_dir, exist_ok=True)
    results = []

    # 读取所有的scenarios，都要做对照实验
    for scenario in config.scenarios:
        # 这里容易搞混；请记住，下面的四个eval是对应同一个算法的四个变种。
        # 因此，面对某个扰动的环境，有四个checkpoint会被测试：在原环境训练、在angle环境训练、没有obs、有obs
        # 当环境定义为raw的时候，只测试前两个变种
        # raw, angle
        # + !scenario.is_raw -> obs, no_obs
        raw_results = eval(
            config,
            checkpoint=checkpoint,
            checkpoint_type="raw",
            eval_scenario=scenario,
        )
        results.append(raw_results)

        angle_results = eval(
            config,
            checkpoint=checkpoint,
            checkpoint_type="angle",
            eval_scenario=scenario,
        )
        results.append(angle_results)

        if not scenario.is_raw:
            obs_results = eval(
                config,
                checkpoint=checkpoint,
                checkpoint_type=f"disturb_{scenario.name}_obs",
                eval_scenario=scenario,
            )
            results.append(obs_results)

            no_obs_results = eval(
                config,
                checkpoint=checkpoint,
                checkpoint_type=f"disturb_{scenario.name}_no_obs",
                eval_scenario=scenario,
            )
            results.append(no_obs_results)

            # 以下是baseline或者说消融；把obs的checkpoint在没有扰动的环境上测试
            _sc = deepcopy(scenario)
            _sc.is_raw = True
            _sc.name = "raw"
            _sc.disturbances = None
            obs_raw_results = eval(
                config,
                checkpoint=checkpoint,
                checkpoint_type=f"disturb_{scenario.name}_obs",
                eval_scenario=_sc,
            )
            results.append(obs_raw_results)
            no_obs_raw_results = eval(
                config,
                checkpoint=checkpoint,
                checkpoint_type=f"disturb_{scenario.name}_no_obs",
                eval_scenario=_sc,
            )
            results.append(no_obs_raw_results)

    return results


def eval(
    globalConfig: hydra_type.EvalConfig,
    checkpoint: hydra_type.CheckpointConfig,
    checkpoint_type: str,
    eval_scenario: hydra_type.ScenarioConfig,
):
    start_time = time.time()
    base_checkpoint_path = f"./results/pettingzoo_mw/multiwalker/{checkpoint.algo}/[{checkpoint.algo}]<{checkpoint_type}>_{checkpoint.timestamp}"
    seed_folder = next(
        folder
        for folder in os.listdir(base_checkpoint_path)
        if folder.startswith("seed-")
    )
    checkpoint_path = os.path.join(base_checkpoint_path, seed_folder, "models")

    # 1. 先读取对应的模型
    rich.print(
        Panel(
            f"Checkpoint Path: {checkpoint_path}\nScenario Name: {eval_scenario.name}",
            title="Evaluation Info",
        )
    )

    # 1.1. 从配置里读取参数，转换为harl使用的格式
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"algorithm={checkpoint.algo}",
                f"environment={checkpoint_type}",
            ],
        )
        algo_args = cfg.algorithm
        env_args = cfg.environment

        algorithm_name = cfg.algorithm.name
        env_name = cfg.environment.name
        scenario_name = cfg.environment.scenario
        basic_info = {
            "env": env_name,
            "algo": algorithm_name,
            "exp_name": f"testing_<{algorithm_name}>_{scenario_name}",
        }
        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("episode_length") is not None
        ):
            algo_args.train.episode_length = 500

        algo_args.train.model_dir = checkpoint_path  # 读取模型！

        # 配置eval遍数
        algo_args.eval.n_eval_rollout_threads = 500
        algo_args.eval.eval_episodes = globalConfig.run.eval_episodes

        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("num_env_steps") is not None
        ):
            algo_args.train.num_env_steps = 1

        env_args.max_cycles = max_cycles

        algo_dict = _to_dict(algo_args)
        del algo_dict["name"]

        env_dict = _to_dict(env_args)
        del env_dict["name"]
        del env_dict["scenario"]

        runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

        # runner.run()

        logger: PettingZooMWLogger = runner.logger
        logger.is_testing = (
            True  # 标识目前在eval；但是eval这个词被它用了，只能用test了。
        )
        runner.eval()

        # 开始计算
        # 2.1 计算提前摔倒的次数
        terminate_cnt = 0
        terminate_arr = logger.test_data["terminate_at"]
        for i in range(len(terminate_arr)):
            if terminate_arr[i] + 2 < max_cycles:  # +2 去除一点边际问题
                terminate_cnt += 1

        end_time = time.time()
        print(
            f"处理[{checkpoint.algo}]<{checkpoint_type}>_{eval_scenario.name} 耗时: {end_time - start_time:.2f}秒"
        )
        return {
            "desc": f"[{checkpoint.algo}]<{checkpoint_type}>_{eval_scenario.name}",
            "algo": checkpoint.algo,
            "variant": checkpoint_type,
            "scenario": eval_scenario.name,
            "terminate_cnt": terminate_cnt,
            "angle_data": logger.test_data["angle_data"],
        }


def save_eval_results(results: dict, output_dir: str, name: str) -> str:
    """保存评估结果到JSON文件"""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{name}.json"
    filepath = os.path.join(output_dir, filename)

    with open(filepath, "w") as f:
        json.dump(results, f)
    return filepath


def load_eval_results(filepath: str) -> dict:
    """从JSON文件加载评估结果,并保存副本到hydra输出目录"""
    with open(filepath, "r") as f:
        data = json.load(f)

    # 保存副本到hydra输出目录
    output_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "data_used"
    )
    os.makedirs(output_dir, exist_ok=True)
    backup_path = os.path.join(output_dir, os.path.basename(filepath))
    with open(backup_path, "w") as f:
        json.dump(data, f)

    return data


def analyze_eval_results(
    results,
    config: hydra_type.EvalConfig,
    checkpoint: hydra_type.CheckpointConfig,
):
    wandb_results = []
    for res in results:
        angle_intervals = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 5),
            (5, 8),
            (8, 10),
            (10, 15),
            (15, float("inf")),
        ]
        baseline_interval_counts = {
            f"angle-{start}-{end}": sum(
                1 for a in res["angle_data"] if start <= abs(a) < end
            )
            for start, end in angle_intervals
        }
        wandb_item = {
            "algo": res["algo"],
            "variant": res["variant"],
            "scenario": res["scenario"],
            "terminate_cnt": res["terminate_cnt"],
            "angle_data": baseline_interval_counts,
        }

        wandb_results.append(wandb_item)


@hydra.main(
    config_path="./eval_configs",
    config_name="config",
    version_base=None,
)
def main(cfg: hydra_type.SettingConfig):
    # 用于json存储的目录
    json_dir = "./eval_results"
    os.makedirs(json_dir, exist_ok=True)
    timestamp = time.strftime("%m%d-%H:%M")
    GlobalHydra.instance().clear()
    # 初始化wandb
    run = wandb.init(
        project="harl-eval",
        name=cfg.setting.run.run_name + "_" + timestamp,
        config=_to_dict(cfg),
        save_code=True,
    )

    def process_checkpoint(checkpoint: hydra_type.CheckpointConfig):
        print(f"Processing checkpoint: {checkpoint.algo} - {checkpoint.desc}")
        should_load_results = cfg.setting.run.load_results
        if should_load_results:
            # 加载已有结果模式
            result_file_name = cfg.setting.run.result_file_name
            if result_file_name == "latest":
                # 从latest子目录加载latest版本
                latest_dir = os.path.join(json_dir, "latest")
                results = load_eval_results(
                    os.path.join(latest_dir, f"{checkpoint.algo}.json")
                )
            else:
                # 从时间戳子目录加载指定版本
                timestamp_dir = os.path.join(json_dir, result_file_name)
                results = load_eval_results(
                    os.path.join(timestamp_dir, f"{checkpoint.algo}.json")
                )
        else:
            # 执行评估模式
            results = run_evaluations(cfg.setting, checkpoint)

            # 创建时间戳子目录
            timestamp_dir = os.path.join(json_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            # 创建latest子目录
            latest_dir = os.path.join(json_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            # 保存结果到时间戳子目录
            save_eval_results(
                results,
                timestamp_dir,
                f"{checkpoint.algo}",
            )

            save_eval_results(results, latest_dir, f"{checkpoint.algo}")

        # 分析结果并获取图表
        wandb_results = []
        analyze_eval_results(results, cfg.setting, checkpoint)
        log_wandb()

    for checkpoint in cfg.setting.checkpoints:
        start_time = time.time()
        process_checkpoint(checkpoint)
        end_time = time.time()
        print(f"{checkpoint.algo} 耗时: {end_time - start_time:.2f}秒")

    # 发送完成通知
    requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/Eval完成")

    run.finish()


if __name__ == "__main__":
    main()
