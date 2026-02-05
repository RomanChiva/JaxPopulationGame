import os
# Make jax use CPU only
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import hydra
from omegaconf import OmegaConf, DictConfig
import os
import matplotlib.pyplot as plt

# Import Shared Training Generator
from algorithms.shared.train import make_train as make_shared_train

# Import Algorithm Modules (Interfaces)
import algorithms.ppo.interface as ppo_interface
import algorithms.online_q.interface as q_interface
import algorithms.online_sarsa.interface as sarsa_interface
import algorithms.average_batch_q.interface as batch_q_interface
import algorithms.wolf.interface as wolf_interface

# Import Defaults
from algorithms.ppo.config.default import get_config as ppo_cfg
from algorithms.online_q.config.default import get_config as q_cfg
from algorithms.online_sarsa.config.default import get_config as sarsa_cfg
from algorithms.average_batch_q.config.default import get_config as batch_q_cfg
from algorithms.wolf.config.default import get_config as wolf_cfg

def get_algo_module(algo_name):
    if algo_name == "ppo": return ppo_interface
    if algo_name == "online_q": return q_interface
    if algo_name == "online_sarsa": return sarsa_interface
    if algo_name == "average_batch_q": return batch_q_interface
    if algo_name == "wolf": return wolf_interface
    raise ValueError(f"Unknown algorithm: {algo_name}")

def get_default_algo_config(algo_name):
    if algo_name == "ppo": return ppo_cfg()
    if algo_name == "online_q": return q_cfg()
    if algo_name == "online_sarsa": return sarsa_cfg()
    if algo_name == "average_batch_q": return batch_q_cfg()
    if algo_name == "wolf": return wolf_cfg()
    return {}

@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    # 1. Merge Configs
    base_algo_cfg = get_default_algo_config(cfg.ALGO)
    final_config = base_algo_cfg.to_dict()
    final_config.update(OmegaConf.to_container(cfg))
    
    # 3. Create Shared Train Function with injected algo module
    algo_module = get_algo_module(cfg.ALGO)
    train_fn = make_shared_train(final_config, algo_module)
    
    # 4. Parallel Execution
    rng = jax.random.PRNGKey(cfg.SEED)
    seeds = jax.random.split(rng, cfg.NUM_SEEDS)
    
    print(f"Compiling {cfg.ALGO} for {cfg.NUM_SEEDS} parallel runs...")
    vmapped_train = jax.vmap(train_fn)
    compiled_train = jax.jit(vmapped_train)
    
    # 5. Run
    print("Starting training...")
    results = compiled_train(seeds)
    print("Training finished.")
    
    # 6. Process Metrics
    metrics = results
    print(metrics.keys())
    print(metrics['cooperation_rate'].shape)  # (NUM_SEEDS, EPISODE_LENGTH)
   
   # Plot cooperation rate for all seeds
    plt.figure(figsize=(10, 6))
    for i in range(cfg.NUM_SEEDS):
        plt.plot(metrics['cooperation_rate'][i], label=f'Seed {i}')
    plt.xlabel('Episode Step')
    plt.ylabel('Cooperation Rate')
    plt.title(f'Cooperation Rate over Time for {cfg.ALGO}')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
