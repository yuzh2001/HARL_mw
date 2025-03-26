"""Train an algorithm."""
import rich.pretty
import wandb
import hydra
from omegaconf import DictConfig
import omegaconf
import rich

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    rich.pretty.pprint(cfg, expand_all=True)

    algo_args = cfg.algorithm
    env_args = cfg.environment

    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.environment.scenario


    from datetime import datetime
    ts = datetime.now().strftime("%m%d-%H%M")
    run_name = f"[{algorithm_name}]<{scenario_name}>_{ts}"

    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }

    if env_name == 'pettingzoo_mw' and algo_args.train.get('episode_length') is not None:
        algo_args.train.episode_length = 500
    
    def _to_dict(cfg1: DictConfig):
        return omegaconf.OmegaConf.to_container(
            cfg1, resolve=True, throw_on_missing=True
        )
    wandb.init(project="HARL", 
               config=_to_dict(cfg), 
               sync_tensorboard=True, 
               name=run_name
               )
    # start training
    from harl.runners import RUNNER_REGISTRY
    
    algo_dict = _to_dict(algo_args)
    del algo_dict["name"]
    env_dict = _to_dict(env_args)
    del env_dict["name"]
    del env_dict["scenario"]

    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)
    runner.run()
    runner.close()
    import requests

    requests.get(f"https://api.day.app/Ya5CADvAuDWf5NR4E8ZGt5/{run_name}训练完成")
    wandb.finish()


if __name__ == "__main__":
    main()
