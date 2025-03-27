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
    columns = ["desc", "env", "reward", "angle", "angle_deg", "steps", "terminate_cnt"]
    # 将字典数据转换为列表格式
    table_data = []
    for result in wandb_results:
        table_data.append(
            [
                result["desc"],
                result["env"],
                result["reward"],
                result["angle"],
                result["angle_deg"],
                result["steps"],
                result["terminate_cnt"],
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
                f"environment={eval_scenario.name}",
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
        algo_args.eval.n_eval_rollout_threads = 1
        algo_args.eval.eval_episodes = globalConfig.run.eval_episodes

        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("num_env_steps") is not None
        ):
            algo_args.train.num_env_steps = 1

        algo_dict = _to_dict(algo_args)
        del algo_dict["name"]

        env_dict = _to_dict(env_args)
        del env_dict["name"]
        del env_dict["scenario"]

        runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

        # runner.run()
        runner.eval()

        exit()


# def old_eval(
#     cfg: EvalConfig,
#     config_name: str,
#     disturbances: list[DisturbanceConfig],
#     checkpoint: CheckpointPath,
#     render_mode: str | None = "rgb_array",
# ):
#     gif_save_path = os.path.join(
#         hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
#     )
#     save_gif = cfg.render.use_gif
#     # 读取模型
#     model = PPO.load(os.path.join("./checkpoint_models", checkpoint.path))

#     # 读取扰动参数
#     rich.print(checkpoint)
#     use_angle_reward = checkpoint.get("use_angle_reward", False)
#     use_f_obs = (
#         checkpoint.get("use_f_obs", False) or model.observation_space.shape[0] == 96
#     )
#     use_motor_obs = checkpoint.get("use_motor_obs", False)
#     use_package_mass_obs = checkpoint.get("use_package_mass_obs", False)

#     SHOULD_NOT_RANDOM_DISTURBANCE = True  # 在测试的时候不应该再允许千分之二的那个进行了
#     use_f_disturbance = (
#         checkpoint.get("use_f_disturbance", False) and len(disturbances) > 0
#     )
#     use_motor_disturbance = checkpoint.get("use_motor_disturbance", False)
#     use_package_mass_disturbance = checkpoint.get("use_package_mass_disturbance", False)

#     if SHOULD_NOT_RANDOM_DISTURBANCE:
#         use_f_disturbance = False
#         use_motor_disturbance = False
#         use_package_mass_disturbance = False

#     # 用于存储所有进程的结果
#     from joblib import Parallel, delayed

#     rewards_out = [0, 0, 0]

#     def run_episode(episode_idx: int):
#         # 每个进程创建独立环境
#         env, raw_env = multiwalker_v9.env_with_raw(
#             use_f_obs=use_f_obs,
#             use_f_disturbance=use_f_disturbance,
#             use_angle_reward=use_angle_reward,
#             use_motor_disturbance=use_motor_disturbance,
#             use_motor_obs=use_motor_obs,
#             use_package_mass_obs=use_package_mass_obs,
#             use_package_mass_disturbance=use_package_mass_disturbance,
#             render_mode=render_mode,
#             n_walkers=5,
#             max_cycles=max_cycles,
#         )
#         env = ss.black_death_v3(env)
#         env = ss.frame_stack_v1(env, 3)

#         # 获取基础环境
#         base_env = raw_env.get_raw_env()
#         if base_env is None:
#             print("base_env is None")
#             base_env = MultiWalkerEnv()

#         # 引入扰动
#         turbances_array = []
#         for disturbance in disturbances:
#             turbances_array.append(DisturbanceFactory(base_env, **disturbance))
#         if len(turbances_array) > 0:
#             raw_env.set_disturbances(turbances_array)

#         rewards = 0
#         episode_frames = []
#         episode_rewards_curve = []

#         # 使用独立的seed
#         env.reset(seed=cfg.seed + episode_idx)

#         if len(turbances_array) > 0:
#             raw_env.set_disturbances(turbances_array)

#         episode_data = {
#             "angles_abs": [],
#             "angles_deg": [],
#             "reward": 0,
#             "steps": 0,
#             "last_episode_angles": [],
#             "last_episode_rewards": {agent: 0 for agent in env.agents},
#         }

#         step = 0
#         terminated = False

#         for agent in env.agent_iter():
#             step += 1
#             obs, reward, termination, truncation, info = env.last()
#             episode_rewards_curve.append(base_env.rewards_group)
#             _r = 0
#             for a in env.agents:
#                 _r += env.rewards[a]
#                 episode_data["reward"] += env.rewards[a]
#             rewards += _r / 3
#             if termination or truncation:
#                 terminated = True
#                 break
#             else:
#                 act = model.predict(obs, deterministic=True)[0]

#             curr_angle = base_env.package.angle
#             episode_data["angles_abs"].append(abs(curr_angle))
#             episode_data["angles_deg"].append(curr_angle / 3.14 * 180)

#             if episode_idx == cfg.eval_episodes - 1:
#                 r_array = env.render()
#                 episode_frames.append(r_array)
#                 episode_data["last_episode_angles"].append(curr_angle / 3.14 * 180)
#                 for a in env.agents:
#                     episode_data["last_episode_rewards"][a] += env.rewards[a]

#             env.step(act)
#         env.reset()
#         episode_data["steps"] = step // 3
#         env.close()

#         return {
#             "episode_data": episode_data,
#             "frames": episode_frames if episode_idx == cfg.eval_episodes - 1 else [],
#             "rewards_curve": episode_rewards_curve,
#             "terminated": terminated,
#             "rewards": rewards,
#         }

#     # 使用joblib并行执行episodes
#     # verbose = 100
#     results = Parallel(n_jobs=n_jobs, backend="loky")(
#         delayed(run_episode)(i) for i in range(cfg.eval_episodes)
#     )

#     # 整理结果
#     episodes_data = []
#     frames = []
#     rewards_curve = []
#     terminate_cnt = 0

#     for result in results:
#         episodes_data.append(result["episode_data"])
#         frames.extend(result["frames"])
#         rewards_curve.extend(result["rewards_curve"])
#         if result["episode_data"]["steps"] < max_cycles:
#             terminate_cnt += 1
#         rewards_out += result["rewards"]

#     # 保存gif
#     timestamp_str = time.strftime("%Y%m%d-%H%M%S")
#     if save_gif and len(frames) > 0:
#         export_gif(
#             config_name,
#             frames,
#             gif_save_path,
#             timestamp_str,
#         )

#     # 计算指标
#     all_angles_abs = [angle for ep in episodes_data for angle in ep["angles_abs"]]

#     # 计算平均奖励
#     avg_reward = rewards_out[0] / cfg.eval_episodes
#     avg_episode_reward = sum(episodes_data[-1]["last_episode_rewards"].values()) / len(
#         episodes_data[-1]["last_episode_rewards"].values()
#     )

#     avg_angle = sum(all_angles_abs) / len(all_angles_abs)
#     avg_angle_deg = avg_angle / 3.14 * 180
#     avg_steps = sum(ep["steps"] for ep in episodes_data) / cfg.eval_episodes

#     return {
#         "reward": avg_reward,
#         "angle": avg_angle,
#         "angle_deg": avg_angle_deg,
#         "steps": avg_steps,
#         "episode_angles": episodes_data[-1]["last_episode_angles"],
#         "angles_abs": all_angles_abs,
#         "angles_deg": [angle for ep in episodes_data for angle in ep["angles_deg"]],
#         "episode_rewards": avg_episode_reward,
#         "terminate_cnt": terminate_cnt,
#         "episodes_data": episodes_data,
#         "rewards_curve": [reward_list.tolist() for reward_list in rewards_curve],
#     }


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
        sync_tensorboard=True,
    )

    def process_checkpoint(checkpoint: hydra_type.CheckpointConfig):
        print(f"Processing checkpoint: {checkpoint.algo} - {checkpoint.desc}")
        should_load_results = cfg.setting.run.load_results
        if should_load_results:
            # 加载已有结果模式
            result_file_name = cfg.setting.run.result_file_name
            # if result_file_name == "latest":
            #     # 从latest子目录加载latest版本
            #     latest_dir = os.path.join(json_dir, "latest")
            #     baseline_results = load_eval_results(
            #         os.path.join(latest_dir, f"{checkpoint.desc}_baseline.json")
            #     )
            #     disturb_results = load_eval_results(
            #         os.path.join(latest_dir, f"{checkpoint.desc}_disturb.json")
            #     )
            # else:
            #     # 从时间戳子目录加载指定版本
            #     timestamp_dir = os.path.join(json_dir, result_file_name)
            #     baseline_results = load_eval_results(
            #         os.path.join(timestamp_dir, f"{checkpoint.desc}_baseline.json")
            #     )
            #     disturb_results = load_eval_results(
            #         os.path.join(timestamp_dir, f"{checkpoint.desc}_disturb.json")
            #     )
        else:
            # 执行评估模式
            run_evaluations(cfg.setting, checkpoint)

            # 创建时间戳子目录
            timestamp_dir = os.path.join(json_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            # 创建latest子目录
            latest_dir = os.path.join(json_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

        #     # 保存结果到时间戳子目录
        #     save_eval_results(
        #         baseline_results,
        #         timestamp_dir,
        #         f"{checkpoint.desc}_baseline",
        #     )
        #     save_eval_results(
        #         disturb_results, timestamp_dir, f"{checkpoint.desc}_disturb"
        #     )

        #     # 同时保存latest版本到latest子目录
        #     save_eval_results(
        #         baseline_results, latest_dir, f"{checkpoint.desc}_baseline"
        #     )
        #     save_eval_results(disturb_results, latest_dir, f"{checkpoint.desc}_disturb")

        # # 分析结果并获取图表
        # analyze_eval_results(baseline_results, disturb_results, cfg, checkpoint)

    for checkpoint in cfg.setting.checkpoints:
        start_time = time.time()
        process_checkpoint(checkpoint)
        end_time = time.time()
        print(f"{checkpoint.algo} 耗时: {end_time - start_time:.2f}秒")

    # 发送完成通知
    # requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/Eval完成")

    log_wandb()
    run.finish()


if __name__ == "__main__":
    main()
