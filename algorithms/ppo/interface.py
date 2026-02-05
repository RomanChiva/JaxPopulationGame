
import jax
import jax.numpy as jnp
from flax import struct
from algorithms.ppo.ppo import ActorCritic, Transition, create_agent_state as ppo_create_state, calculate_gae, ppo_update
from algorithms.shared.train import Experience

# PPO State no longer needs buffer
@struct.dataclass
class PPOAgentState:
    train_state: any

def init(key, config, obs_shape, num_actions):
    ppo_ts = ppo_create_state(key, config, obs_shape, num_actions)
    return PPOAgentState(train_state=ppo_ts)

def select_action(agent_state, obs, key, config):
    pi, value = agent_state.train_state.apply_fn(agent_state.train_state.params, obs)
    action = pi.sample(seed=key)
    log_prob = pi.log_prob(action)
    return action, (value, log_prob)

def update(agent_state, batch_experience: Experience, key, config):
   
    value, log_prob = batch_experience.extras
    
  
    traj = Transition(
        done=batch_experience.done,
        action=batch_experience.action,
        value=value,
        reward=batch_experience.reward,
        log_prob=log_prob,
        obs=batch_experience.obs
    )
  
    eps, steps = batch_experience.obs.shape[0], batch_experience.obs.shape[1]
    
    # Only compute bootstrap values for last next_obs
    last_next_obs = batch_experience.next_obs[:, -1, :]  # Shape: (eps, obs_dim)
    _, bootstrap_vals = agent_state.train_state.apply_fn(
        agent_state.train_state.params, 
        last_next_obs)
    
    vmapped_gae = jax.vmap(calculate_gae, in_axes=(0, 0, None))
    advantages, targets = vmapped_gae(traj, bootstrap_vals, config)
    
    flat_traj = jax.tree.map(lambda x: x.reshape((eps * steps,) + x.shape[2:]), traj)
    flat_adv = advantages.reshape((eps * steps,))
    flat_tgt = targets.reshape((eps * steps,))
    
    # Update
    new_ts, metrics = ppo_update(
        agent_state.train_state, 
        flat_traj, 
        flat_adv, 
        flat_tgt, 
        key, 
        config
    )
    
    return PPOAgentState(train_state=new_ts), metrics
