
import jax
import jax.numpy as jnp
from env import PopulationEnv
from flax import struct
from functools import partial

@struct.dataclass
class Experience:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    next_obs: jnp.ndarray
    extras: any 


def make_train(config, algo_module):
    
    def train(rng):
        # 1. INITIALIZE ENVIRONMENT
        env = PopulationEnv(
            num_agents=config["NUM_AGENTS"],
            norm_string=config["NORM_STRING"],
            b=config["B_BENEFIT"],
            c=config["C_COST"],
            max_steps_per_episode=config["EPISODE_LENGTH"],
            obs_type=config["OBS_TYPE"]
        )
        
        # 2. INITIALIZE POPULATION
        rng, init_rng = jax.random.split(rng)
        init_keys = jax.random.split(init_rng, config["NUM_AGENTS"])
        def _init_single(k): return algo_module.init(k, config, env.observation_shape, env.num_actions)
        population_states = jax.vmap(_init_single)(init_keys)

        # 3. INITIALIZE ENV STATE
        # We perform an initial reset to get valid state/obs
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)

        # 4. PREPARE VMAPS
        # partial config
        select_fn = partial(algo_module.select_action, config=config)
        update_fn = partial(algo_module.update, config=config)
        
        vmapped_select = jax.vmap(select_fn, in_axes=(0, 0, 0))
        vmapped_update = jax.vmap(update_fn, in_axes=(0, 0, 0))

        # --- ONLINE TRAINING LOOP ---
        def _run_online_training(initial_pop_state, initial_env_state, initial_obs, run_rng):
            
            
            def _episode_step(episode_carry, unused):
                pop_states, env_state, current_obs, rng = episode_carry
                
                rng, reset_rng = jax.random.split(rng)
                current_obs, env_state = env.reset(reset_rng)
                
                def _env_step(carry, unused):
                    pop, estate, c_obs, rng = carry
                    
                    step_key, rng = jax.random.split(rng)
                    act_rng, env_rng, update_rng = jax.random.split(step_key, 3)
                    act_keys = jax.random.split(act_rng, config["NUM_AGENTS"])
                    
                    actions, extras = vmapped_select(pop, c_obs, act_keys)
                    
                    next_obs, next_estate, rewards, dones, info = env.step(estate, actions, env_rng)
                    
                    experience = Experience(c_obs, actions, rewards, dones, next_obs, extras)
                    
                    update_keys = jax.random.split(update_rng, config["NUM_AGENTS"])
                    updated_pop, algo_metrics = vmapped_update(pop, experience, update_keys)
                    
                    stats = {
                        "avg_reward": info["avg_reward"],
                        "cooperation_rate": info["cooperation_rate"],
                        "avg_reputation": info["good_reputation_rate"]
                    }
                    stats.update(algo_metrics)
                    
                    return (updated_pop, next_estate, next_obs, rng), stats

                # Scan steps
                (final_pop, final_estate, final_obs, rng), step_metrics = jax.lax.scan(
                    _env_step, (pop_states, env_state, current_obs, rng), None, config["EPISODE_LENGTH"]
                )
                
                # Average metrics over episode
                episode_metrics = jax.tree.map(jnp.mean, step_metrics)
                
                return (final_pop, final_estate, final_obs, rng), episode_metrics

            initial_carry = (initial_pop_state, initial_env_state, initial_obs, run_rng)

            (final_state, _, _, _), metrics = jax.lax.scan(_episode_step, initial_carry, None, config["TOTAL_EPISODES"])
            return metrics

        # --- BATCH TRAINING LOOP ---
        def _run_batch_training(initial_pop_state, initial_env_state, initial_obs, run_rng):
            
            # We scan over NUM_UPDATES
            episodes_per_update = config["EPISODES_PER_UPDATE"]
            num_updates = config["TOTAL_EPISODES"] // episodes_per_update
            
            def _update_step(update_carry, unused):
                pop_states, env_state, current_obs, rng = update_carry
                
                # Collect N episodes
                def _collect_episode(ep_carry, unused):
                    pop, estate, c_obs, rng = ep_carry
                    
                    rng, reset_rng = jax.random.split(rng)
                    c_obs, estate = env.reset(reset_rng)
                    
                    def _collect_step(step_carry, unused):
                        p, es, co, r = step_carry
                        
                        r, step_r = jax.random.split(r)
                        act_r, env_r = jax.random.split(step_r, 2)
                        act_ks = jax.random.split(act_r, config["NUM_AGENTS"])
                        
                        actions, extras = vmapped_select(p, co, act_ks)
                        no, nes, rews, d, info = env.step(es, actions, env_r)
                        
                        exp = Experience(co, actions, rews, d, no, extras)
                        
                        # Return exp and metrics
                        stats = {
                            "avg_reward": info["avg_reward"],
                            "cooperation_rate": info["cooperation_rate"],
                            "avg_reputation": info["good_reputation_rate"]
                        }
                        return (p, nes, no, r), (exp, stats)
                    
                    # Scan steps
                    (pop, estate, c_obs, rng), (traj, stats) = jax.lax.scan(
                        _collect_step, (pop, estate, c_obs, rng), None, config["EPISODE_LENGTH"]
                    )
                    return (pop, estate, c_obs, rng), (traj, stats)

                # Scan episodes
                (pop_states, env_state, current_obs, rng), (batch_traj, batch_stats) = jax.lax.scan(
                    _collect_episode, (pop_states, env_state, current_obs, rng), None, episodes_per_update
                )
                
                batch_traj_swapped = jax.tree.map(lambda x: jnp.moveaxis(x, 2, 0), batch_traj)
                
                rng, update_rng = jax.random.split(rng)
                update_keys = jax.random.split(update_rng, config["NUM_AGENTS"])
                

                # Update function for Batch/PPO receives the full batch
                updated_pop, algo_metrics = vmapped_update(pop_states, batch_traj_swapped, update_keys)
                
                # Aggregate/Mean metrics
                # batch_stats: (EPISODES, STEPS)
                mean_stats = jax.tree.map(jnp.mean, batch_stats)
                mean_stats.update(algo_metrics)
                
                return (updated_pop, env_state, current_obs, rng), mean_stats
            
            initial_carry = (initial_pop_state, initial_env_state, initial_obs, run_rng)

            (final_state, _, _, _), metrics = jax.lax.scan(_update_step, initial_carry, None, num_updates)
            return metrics

        # DISPATCH
        if config["ALGO"] in ["ppo", "average_batch_q"]:
            return _run_batch_training(population_states, env_state, obs, rng)
        else:
            return _run_online_training(population_states, env_state, obs, rng)

    return train
