
import jax
import jax.numpy as jnp
from algorithms.online_q.online_q import create_agent_state, select_action as q_select, update as q_update

def init(key, config, obs_shape, num_actions):
    obs_dim = obs_shape[0]
    if obs_dim == 1: num_states = 2 
    elif obs_dim == 2: num_states = 4
    elif obs_dim == 3: num_states = 40
    else: num_states = 4
    return create_agent_state(key, num_states, num_actions, config)

def select_action(agent_state, obs, key, config):
    action = q_select(agent_state, obs, key, config["EPSILON"])
    return action, () # Extras empty

def update(agent_state, experience, key, config):
    # q_update(state, obs, action, r, next_obs, done, lr, gamma)
    new_state = q_update(
        agent_state,
        experience.obs,
        experience.action,
        experience.reward,
        experience.next_obs,
        experience.done,
        config["LR"],
        config["DISCOUNT"]
    )
    # Metrics
    metrics = {} # Q learning doesn't return loss usually unless computed
    return new_state, metrics
