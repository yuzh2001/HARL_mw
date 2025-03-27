from __future__ import annotations

import json
import os
import time

import hydra
import matplotlib.pyplot as plt
import numpy as np
import rich
import supersuit as ss
from matplotlib import font_manager
from omegaconf import DictConfig
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from stable_baselines3 import PPO

import wandb
from harl.envs.pettingzoo_mw.walker.disturbances import (
    DisturbanceFactory,
    MultiWalkerEnv,
)
from utils.gif import export_gif
from walker import multiwalker_v9


from rich.progress import Progress

os.environ["SDL_VIDEODRIVER"] = "dummy"

# 设置字体路径
font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 确保路径正确
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

max_cycles = 500
n_jobs = 4

wandb_results = []


def eval(
    cfg: DictConfig,
    config_name: str,
    turbances_yaml: list[dict],
    cp: DictConfig,
    gif_save_path: str,
    render_mode: str | None = "rgb_array",
    config_data: dict = None,
    save_gif: bool = False,
):
    # 读取模型
    model = PPO.load(os.path.join("./checkpoint_models", cp.path))

    # 读取扰动参数
    rich.print(cp)
    use_angle_reward = cp.get("use_angle_reward", False)
    use_f_obs = cp.get("use_f_obs", False) or model.observation_space.shape[0] == 96
    use_motor_obs = cp.get("use_motor_obs", False)
    use_package_mass_obs = cp.get("use_package_mass_obs", False)

    SHOULD_NOT_RANDOM_DISTURBANCE = True
    use_f_disturbance = cp.get("use_f_disturbance", False) and len(turbances_yaml) > 0
    use_motor_disturbance = cp.get("use_motor_disturbance", False)
    use_package_mass_disturbance = cp.get("use_package_mass_disturbance", False)

    if SHOULD_NOT_RANDOM_DISTURBANCE:
        use_f_disturbance = False
        use_motor_disturbance = False
        use_package_mass_disturbance = False

    # 用于存储所有进程的结果
    from joblib import Parallel, delayed

    rewards_out = [0, 0, 0]

    def run_episode(episode_idx: int):
        # 每个进程创建独立环境
        env, raw_env = multiwalker_v9.env_with_raw(
            use_f_obs=use_f_obs,
            use_f_disturbance=use_f_disturbance,
            use_angle_reward=use_angle_reward,
            use_motor_disturbance=use_motor_disturbance,
            use_motor_obs=use_motor_obs,
            use_package_mass_obs=use_package_mass_obs,
            use_package_mass_disturbance=use_package_mass_disturbance,
            render_mode=render_mode,
            n_walkers=5,
            max_cycles=max_cycles,
        )
        env = ss.black_death_v3(env)
        env = ss.frame_stack_v1(env, 3)

        # 获取基础环境
        base_env = raw_env.get_raw_env()
        if base_env is None:
            print("base_env is None")
            base_env = MultiWalkerEnv()

        # 引入扰动
        turbances_array = []
        for disturbance in turbances_yaml:
            turbances_array.append(DisturbanceFactory(base_env, **disturbance))
        if len(turbances_array) > 0:
            raw_env.set_disturbances(turbances_array)

        rewards = 0
        episode_frames = []
        episode_rewards_curve = []

        # 使用独立的seed
        env.reset(seed=cfg.seed + episode_idx)

        if len(turbances_array) > 0:
            raw_env.set_disturbances(turbances_array)

        episode_data = {
            "angles_abs": [],
            "angles_deg": [],
            "reward": 0,
            "steps": 0,
            "last_episode_angles": [],
            "last_episode_rewards": {agent: 0 for agent in env.agents},
        }

        step = 0
        terminated = False

        for agent in env.agent_iter():
            step += 1
            obs, reward, termination, truncation, info = env.last()
            episode_rewards_curve.append(base_env.rewards_group)
            _r = 0
            for a in env.agents:
                _r += env.rewards[a]
                episode_data["reward"] += env.rewards[a]
            rewards += _r / 3
            if termination or truncation:
                terminated = True
                break
            else:
                act = model.predict(obs, deterministic=True)[0]

            curr_angle = base_env.package.angle
            episode_data["angles_abs"].append(abs(curr_angle))
            episode_data["angles_deg"].append(curr_angle / 3.14 * 180)

            if episode_idx == cfg.eval_episodes - 1:
                r_array = env.render()
                episode_frames.append(r_array)
                episode_data["last_episode_angles"].append(curr_angle / 3.14 * 180)
                for a in env.agents:
                    episode_data["last_episode_rewards"][a] += env.rewards[a]

            env.step(act)
        env.reset()
        episode_data["steps"] = step // 3
        env.close()

        return {
            "episode_data": episode_data,
            "frames": episode_frames if episode_idx == cfg.eval_episodes - 1 else [],
            "rewards_curve": episode_rewards_curve,
            "terminated": terminated,
            "rewards": rewards,
        }

    # 使用joblib并行执行episodes
    # verbose = 100
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(run_episode)(i) for i in range(cfg.eval_episodes)
    )

    # 整理结果
    episodes_data = []
    frames = []
    rewards_curve = []
    terminate_cnt = 0

    for result in results:
        episodes_data.append(result["episode_data"])
        frames.extend(result["frames"])
        rewards_curve.extend(result["rewards_curve"])
        if result["episode_data"]["steps"] < max_cycles:
            terminate_cnt += 1
        rewards_out += result["rewards"]

    # 保存gif
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")
    if save_gif and len(frames) > 0:
        export_gif(
            config_name,
            frames,
            gif_save_path,
            config_data,
            timestamp_str,
        )

    # 计算指标
    all_angles_abs = [angle for ep in episodes_data for angle in ep["angles_abs"]]

    # 计算平均奖励
    avg_reward = rewards_out[0] / cfg.eval_episodes
    avg_episode_reward = sum(episodes_data[-1]["last_episode_rewards"].values()) / len(
        episodes_data[-1]["last_episode_rewards"].values()
    )

    avg_angle = sum(all_angles_abs) / len(all_angles_abs)
    avg_angle_deg = avg_angle / 3.14 * 180
    avg_steps = sum(ep["steps"] for ep in episodes_data) / cfg.eval_episodes

    return {
        "reward": avg_reward,
        "angle": avg_angle,
        "angle_deg": avg_angle_deg,
        "steps": avg_steps,
        "episode_angles": episodes_data[-1]["last_episode_angles"],
        "angles_abs": all_angles_abs,
        "angles_deg": [angle for ep in episodes_data for angle in ep["angles_deg"]],
        "episode_rewards": avg_episode_reward,
        "terminate_cnt": terminate_cnt,
        "episodes_data": episodes_data,
        "rewards_curve": [reward_list.tolist() for reward_list in rewards_curve],
    }


def run_evaluations(cfg: DictConfig, checkpoint_path: dict) -> tuple[dict, dict]:
    """执行baseline和扰动测试的评估"""
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
    )
    os.makedirs(gif_dir, exist_ok=True)
    # 运行baseline评估
    baseline_config = {"task": {"name": "baseline", "seed": 42}, "disturbances": []}
    baseline_results = eval(
        cfg,
        cp=checkpoint_path,
        config_name="[baseline]" + checkpoint_path.desc,
        turbances_yaml=baseline_config["disturbances"],
        gif_save_path=gif_dir,
        render_mode="rgb_array",
        config_data=baseline_config,
        save_gif=cfg.render.use_gif,
    )

    # 运行扰动测试评估
    disturb_results = eval(
        cfg,
        cp=checkpoint_path,
        config_name="[disturbance]" + checkpoint_path.desc,
        turbances_yaml=cfg.disturbances["disturbances"],
        gif_save_path=gif_dir,
        render_mode="rgb_array",
        config_data=cfg.disturbances,
        save_gif=cfg.render.use_gif,
    )

    return baseline_results, disturb_results


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


def _append_wandb_results(results: dict):
    wandb_results.append(results)


def analyze_eval_results(
    baseline_results: dict,
    disturb_results: dict,
    cfg: DictConfig,
    checkpoint_path: dict,
):
    """分析评估结果并生成报告和图表"""
    console = Console()
    log_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    file_console = Console(
        file=open(os.path.join(log_dir, "runtime.log"), "a+", encoding="utf-8")
    )
    # 计算分析指标
    same_count = sum(
        1
        for a, b in zip(
            baseline_results["angles_deg"],
            disturb_results["angles_deg"],
        )
        if a == b
    )
    # 计算角度区间的数量
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
        f"{start}-{end}": sum(
            1 for a in baseline_results["angles_deg"] if start <= abs(a) < end
        )
        for start, end in angle_intervals
    }
    disturb_interval_counts = {
        f"{start}-{end}": sum(
            1 for a in disturb_results["angles_deg"] if start <= abs(a) < end
        )
        for start, end in angle_intervals
    }

    def format_metric_comparison(key, baseline_val, disturb_val, precision=2):
        diff = disturb_val - baseline_val
        color = "green" if diff > 0 else "red"
        diff_str = "+" if diff > 0 else ""
        return f"{baseline_val:.{precision}f}|{disturb_val:.{precision}f} [{color}]({diff_str}{diff:.{precision}f})[/]"

    # 添加结果到wandb
    wandb_results.extend(
        [
            {
                "desc": checkpoint_path.desc,
                "env": "base",
                "reward": baseline_results["reward"],
                "angle": baseline_results["angle"],
                "angle_deg": baseline_results["angle_deg"],
                "steps": baseline_results["steps"],
                "terminate_cnt": baseline_results["terminate_cnt"],
            },
            {
                "desc": checkpoint_path.desc,
                "env": "disturb",
                "reward": disturb_results["reward"],
                "angle": disturb_results["angle"],
                "angle_deg": disturb_results["angle_deg"],
                "steps": disturb_results["steps"],
                "terminate_cnt": disturb_results["terminate_cnt"],
            },
        ]
    )

    # 生成报告
    output = f"""
    [bold cyan]评估结果 - {checkpoint_path.desc}[/]
    
    [bold]平均奖励:[/] {format_metric_comparison("reward", baseline_results["reward"], disturb_results["reward"], 2)}
    [bold]平均弧度波动:[/] {format_metric_comparison("angle", baseline_results["angle"], disturb_results["angle"], 5)}
    [bold]....角度波动:[/] {format_metric_comparison("angle_deg", baseline_results["angle_deg"], disturb_results["angle_deg"], 5)}
    [bold]平均步数:[/] {format_metric_comparison("steps", baseline_results["steps"], disturb_results["steps"], 1)}
    
    [bold]失败摔倒次数:[/] {format_metric_comparison("terminate_cnt", baseline_results["terminate_cnt"], disturb_results["terminate_cnt"], 1)}

    [bold]角度区间统计:[/]
    [bold]0-1 度次数:[/] {format_metric_comparison("angle_0_1", baseline_interval_counts["0-1"], disturb_interval_counts["0-1"], 1)}
    [bold]1-2 度次数:[/] {format_metric_comparison("angle_1_2", baseline_interval_counts["1-2"], disturb_interval_counts["1-2"], 1)}
    [bold]2-3 度次数:[/] {format_metric_comparison("angle_2_3", baseline_interval_counts["2-3"], disturb_interval_counts["2-3"], 1)}
    [bold]3-5 度次数:[/] {format_metric_comparison("angle_3_5", baseline_interval_counts["3-5"], disturb_interval_counts["3-5"], 1)}
    [bold]5-8 度次数:[/] {format_metric_comparison("angle_5_8", baseline_interval_counts["5-8"], disturb_interval_counts["5-8"], 1)}
    [bold]8-10 度次数:[/] {format_metric_comparison("angle_8_10", baseline_interval_counts["8-10"], disturb_interval_counts["8-10"], 1)}
    [bold]10-15 度次数:[/] {format_metric_comparison("angle_10_15", baseline_interval_counts["10-15"], disturb_interval_counts["10-15"], 1)}
    [bold]>15 度次数:[/] {format_metric_comparison("angle_15_inf", baseline_interval_counts["15-inf"], disturb_interval_counts["15-inf"], 1)}
    
    [bold]角度一样次数:[/] {same_count // 3}
    [bold]评估次数:[/] {cfg.eval_episodes}
    """

    panel = Panel(
        output.strip(),
        title="[bold]评估报告",
        title_align="center",
        padding=(1, 4),
        style="blue",
    )

    console.print(panel)
    file_console.print(panel)

    # 绘制图表
    timestamp_str = time.strftime("%Y%m%d-%H%M%S")

    gif_folder = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
        "legends",
        f"{timestamp_str}_{checkpoint_path.desc}",
    )
    os.makedirs(gif_folder, exist_ok=True)

    # 绘制角度分布对比图
    fig, (ax_base, ax_disturb) = plt.subplots(2, 1, figsize=(12, 8))

    # 准备数据
    labels = list(baseline_interval_counts.keys())
    baseline_values = list(baseline_interval_counts.values())
    disturb_values = list(disturb_interval_counts.values())

    # 绘制baseline分布
    ax_base.bar(labels, baseline_values, color="blue", alpha=0.6)
    ax_base.set_title("Baseline角度分布", fontproperties=font_prop)
    ax_base.set_ylabel("cnt")
    ax_base.tick_params(axis="x", rotation=45)

    # 绘制disturb分布
    ax_disturb.bar(labels, disturb_values, color="red", alpha=0.6)
    ax_disturb.set_title("Disturb角度分布", fontproperties=font_prop)
    ax_disturb.set_ylabel("cnt")
    ax_disturb.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(f"{gif_folder}/angle_distribution.png")
    plt.close()

    # 绘制角度对比图
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 4))

    ax1.plot(disturb_results["episode_angles"], label=checkpoint_path.desc)
    ax1.set_title(checkpoint_path.desc, fontproperties=font_prop)
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Angle (rad)")
    ax1.set_xlim(0, 1500)
    ax1.set_ylim(-20, 20)
    ax1.grid(True)
    ax1.legend()
    plt.tight_layout()

    plt.savefig(f"{gif_folder}/angles_comparison.png")
    plt.close()

    file_console.file.close()

    # 绘制奖励曲线
    rewards_curve = disturb_results["rewards_curve"]
    # 现在的rewards_curve是(steps, n_walkers, 3)
    # 需要把它合并为(steps, 3)
    rewards_curve_merge = np.sum(rewards_curve, axis=1)
    rewards_curve_merge[0][1] = 0
    rewards_curve_merge[1][1] = 0
    rewards_curve_merge[2][1] = 0
    fig, ax2 = plt.subplots(1, 1, figsize=(12, 4))
    ax2.plot(rewards_curve_merge[:, 0], label="shaping")
    ax2.plot(rewards_curve_merge[:, 1], label="package")
    ax2.plot(rewards_curve_merge[:, 2], label="angle")
    ax2.plot(rewards_curve_merge[:, 3], label="angle_abs")
    ax2.set_title(checkpoint_path.desc, fontproperties=font_prop)
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Reward")
    ax2.set_ylim(-2, 2)
    ax2.legend()
    plt.savefig(f"{gif_folder}/rewards_comparison.png")
    wandb.log(
        {
            f"{checkpoint_path.desc}/rewards_comparison": wandb.Image(
                f"{gif_folder}/rewards_comparison.png"
            )
        }
    )
    plt.close()

    # 再绘制一个奖励累积曲线
    rewards_curve_cumsum = np.cumsum(rewards_curve_merge, axis=0)
    fig, ax3 = plt.subplots(1, 1, figsize=(12, 4))
    ax3.plot(rewards_curve_cumsum[:, 0], label="shaping")
    ax3.plot(rewards_curve_cumsum[:, 1], label="package")
    ax3.plot(rewards_curve_cumsum[:, 2], label="angle")
    ax3.plot(rewards_curve_cumsum[:, 3], label="angle_abs")
    ax3.set_title(checkpoint_path.desc, fontproperties=font_prop)
    ax3.set_xlabel("Steps")
    ax3.set_ylabel("Reward")
    ax3.legend()
    plt.savefig(f"{gif_folder}/rewards_cumsum_comparison.png")
    wandb.log(
        {
            f"{checkpoint_path.desc}/rewards_cumsum_comparison": wandb.Image(
                f"{gif_folder}/rewards_cumsum_comparison.png"
            )
        }
    )
    plt.close()


@hydra.main(
    config_path="./configs/tests/settings",
    config_name="0117-obs",
    version_base=None,
)
def main(cfg: DictConfig):
    # 用于json存储的目录
    json_dir = "./eval_results"
    os.makedirs(json_dir, exist_ok=True)
    timestamp = time.strftime("%m%d-%H:%M")

    # 初始化wandb
    run = wandb.init(
        project="sb3-eval",
        name=cfg.run_name + "_" + timestamp,
        config=dict(cfg),
        save_code=True,
    )
    wandb.save("src/*", base_path=".")
    wandb.save("src/walker/*", base_path=".")
    wandb.save("src/walker/multiwalker/*", base_path=".")

    all_data = []
    draw_all = False

    def process_checkpoint(checkpoint_path):
        print(f"Processing checkpoint: {checkpoint_path.desc}")
        if cfg.get("load_results", False):  # 从配置中读取是否加载已有结果
            # 加载已有结果模式
            result_file_name = cfg.get("result_file_name", "latest")
            if result_file_name == "latest":
                # 从latest子目录加载latest版本
                latest_dir = os.path.join(json_dir, "latest")
                baseline_results = load_eval_results(
                    os.path.join(latest_dir, f"{checkpoint_path.desc}_baseline.json")
                )
                disturb_results = load_eval_results(
                    os.path.join(latest_dir, f"{checkpoint_path.desc}_disturb.json")
                )
            else:
                # 从时间戳子目录加载指定版本
                timestamp_dir = os.path.join(json_dir, result_file_name)
                baseline_results = load_eval_results(
                    os.path.join(timestamp_dir, f"{checkpoint_path.desc}_baseline.json")
                )
                disturb_results = load_eval_results(
                    os.path.join(timestamp_dir, f"{checkpoint_path.desc}_disturb.json")
                )
        else:
            # 执行评估模式
            baseline_results, disturb_results = run_evaluations(cfg, checkpoint_path)

            # 创建时间戳子目录
            timestamp_dir = os.path.join(json_dir, timestamp)
            os.makedirs(timestamp_dir, exist_ok=True)

            # 创建latest子目录
            latest_dir = os.path.join(json_dir, "latest")
            os.makedirs(latest_dir, exist_ok=True)

            # 保存结果到时间戳子目录
            save_eval_results(
                baseline_results,
                timestamp_dir,
                f"{checkpoint_path.desc}_baseline",
            )
            save_eval_results(
                disturb_results, timestamp_dir, f"{checkpoint_path.desc}_disturb"
            )

            # 同时保存latest版本到latest子目录
            save_eval_results(
                baseline_results, latest_dir, f"{checkpoint_path.desc}_baseline"
            )
            save_eval_results(
                disturb_results, latest_dir, f"{checkpoint_path.desc}_disturb"
            )

        # 分析结果并获取图表
        analyze_eval_results(baseline_results, disturb_results, cfg, checkpoint_path)

        if draw_all:
            # 计算数据
            episodes = range(1, len(disturb_results["episodes_data"]) + 1)
            disturb_max_angles = [
                max(ep["angles_abs"]) * 180 / 3.14
                for ep in disturb_results["episodes_data"]
            ]
            disturb_final_rewards = [
                ep["reward"] for ep in disturb_results["episodes_data"]
            ]
            disturb_avg_angles = [
                sum(ep["angles_abs"]) * 180 / 3.14 / len(ep["angles_abs"])
                for ep in disturb_results["episodes_data"]
            ]
            disturb_final_steps = [
                ep["steps"] for ep in disturb_results["episodes_data"]
            ]

            all_data.append(
                {
                    "desc": checkpoint_path.desc,
                    "episodes": episodes,
                    "max_angles": disturb_max_angles,
                    "rewards": disturb_final_rewards,
                    "avg_angles": disturb_avg_angles,
                    "steps": disturb_final_steps,
                }
            )

    with Progress() as progress:
        task0 = progress.add_task(
            "[green]Checkpoint evaluating...", total=len(cfg.checkpoint_path)
        )
        # 顺序处理每个checkpoint
        for checkpoint_path in cfg.checkpoint_path:
            start_time = time.time()
            process_checkpoint(checkpoint_path)
            progress.update(task0, advance=1)
            end_time = time.time()
            print(
                f"处理checkpoint {checkpoint_path.desc} 耗时: {end_time - start_time:.2f}秒"
            )

    def _draw():
        # 创建一个新的图表用于比较所有checkpoint
        compare_fig, ((cax1, cax2), (cax3, cax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 设置比较图表的标题和范围
        cax1.set_title("Maximum Angle per Episode(abs)")
        cax1.set_xlabel("Episode")
        cax1.set_ylabel("Max Angle (abs)")
        cax1.set_xlim(0, 100)
        cax1.set_ylim(0, 30)

        cax2.set_title("Final Reward per Episode")
        cax2.set_xlabel("Episode")
        cax2.set_ylabel("Reward")
        cax2.set_xlim(0, 100)
        cax2.set_ylim(-300, 400)

        cax3.set_title("Average Angle per Episode")
        cax3.set_xlabel("Episode")
        cax3.set_ylabel("Avg Angle")
        cax3.set_xlim(0, 100)
        cax3.set_ylim(0, 10)

        cax4.set_title("Steps per Episode")
        cax4.set_xlabel("Episode")
        cax4.set_ylabel("Steps")
        cax4.set_xlim(0, 100)
        cax4.set_ylim(0, 600)

        # 一次性绘制所有数据
        for data in all_data:
            cax1.plot(data["episodes"], data["max_angles"], label=data["desc"])
            cax2.plot(data["episodes"], data["rewards"], label=data["desc"])
            cax3.plot(data["episodes"], data["avg_angles"], label=data["desc"])
            cax4.plot(data["episodes"], data["steps"], label=data["desc"])

        # 添加图例
        cax1.legend(prop=font_prop)
        cax2.legend(prop=font_prop)
        cax3.legend(prop=font_prop)
        cax4.legend(prop=font_prop)

        plt.tight_layout()
        plt.gcf().set_size_inches(20, 10)

        # 保存比较图表
        compare_dir = os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "comparison"
        )
        os.makedirs(compare_dir, exist_ok=True)
        plt.savefig(
            os.path.join(compare_dir, "all_checkpoints_comparison.png"),
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    # _draw()
    print(
        f"Hydra config path: {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )

    import requests

    requests.get("https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/Eval完成")

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
    run.finish()


if __name__ == "__main__":
    main()
