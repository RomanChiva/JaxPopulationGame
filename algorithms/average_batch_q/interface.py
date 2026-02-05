
import jax
import jax.numpy as jnp
from flax import struct
from algorithms.average_batch_q.average_batch_q import create_agent_state, select_action as q_select, get_state_index

# Batch Q needs just base state?
# Yes, we accumulate gradients over the PROVIDED batch, then apply.
# We don't need persistent accumulation buffer across updates because the loop provides the full batch.

def init(key, config, obs_shape, num_actions):
    obs_dim = obs_shape[0]
    if obs_dim == 1: num_states = 2
    elif obs_dim == 2: num_states = 4
    elif obs_dim == 3: num_states = 40
    else: num_states = 4
    return create_agent_state(key, num_states, num_actions, config)

def select_action(agent_state, obs, key, config):
    return q_select(agent_state, obs, key, config["EPSILON"]), ()

def update(agent_state, batch_experience, key, config):
    # batch_experience: (EPISODES, STEPS, ...)
    # Flatten to (TOTAL_SAMPLES, ...)
    eps, steps = batch_experience.obs.shape[0], batch_experience.obs.shape[1]
    
    flat_obs = batch_experience.obs.reshape((eps * steps,) + batch_experience.obs.shape[2:])
    flat_act = batch_experience.action.reshape((eps * steps,))
    flat_rew = batch_experience.reward.reshape((eps * steps,))
    flat_next_obs = batch_experience.next_obs.reshape((eps * steps,) + batch_experience.next_obs.shape[2:])
    flat_done = batch_experience.done.reshape((eps * steps,))
    
    # Calculate TD errors for all samples
    # We can vmap over samples?
    
    def _compute_error(curr_obs, act, reward, next_obs, done):
        curr_idx = get_state_index(curr_obs)
        next_idx = get_state_index(next_obs)
        
        best_next_q = jnp.max(agent_state.q_table[next_idx])
        # If done, target is reward? (Standard Q)
        target = reward + config["DISCOUNT"] * best_next_q * (1.0 - done)
        
        curr_q = agent_state.q_table[curr_idx, act]
        return curr_idx, act, (target - curr_q)

    # Vmap over batch
    indices, actions, errors = jax.vmap(_compute_error)(flat_obs, flat_act, flat_rew, flat_next_obs, flat_done)
    
    # Aggregating gradients
    # We need to sum errors for each (state, action) pair
    # jax.ops.segment_sum? Or .at[].add()
    
    grad_acc = jnp.zeros_like(agent_state.q_table)
    count_acc = jnp.zeros_like(agent_state.q_table)
    
    # We can iterate over the batch or use specialized scatter add
    # grad_acc = grad_acc.at[indices, actions].add(errors)
    # count_acc = count_acc.at[indices, actions].add(1)
    
    grad_acc = grad_acc.at[indices, actions].add(errors)
    count_acc = count_acc.at[indices, actions].add(1)
    
    # Apply
    safe_counts = jnp.maximum(count_acc, 1.0)
    avg_grads = grad_acc / safe_counts
    
    # Only update visited
    mask = count_acc > 0
    updates = jnp.where(mask, avg_grads, 0.0)
    
    new_q = agent_state.q_table + config["LR"] * updates
    
    return agent_state.replace(q_table=new_q), {}
