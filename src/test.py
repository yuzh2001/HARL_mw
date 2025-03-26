
import hydra
import time
import torch

@hydra.main(config_path="configs", config_name="config", version_base=None)
def hello(cfg):
  time.sleep(10)
  print(cfg.name)

if __name__ == "__main__":
  hello()

