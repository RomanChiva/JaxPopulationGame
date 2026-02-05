
import jax
import jax.numpy as jnp
from algorithms.wolf.wolf import create_agent_state, select_action as wolf_select, update as wolf_update

def init(key, config, obs_shape, num_actions):
    obs_dim = obs_shape[0]
    if obs_dim == 1: num_states = 2 
    elif obs_dim == 2: num_states = 4
    elif obs_dim == 3: num_states = 40
    else: num_states = 4
    return create_agent_state(key, num_states, num_actions, config)

def select_action(agent_state, obs, key, config):
    action = wolf_select(agent_state, obs, key)
    return action, ()

def update(agent_state, experience, key, config):
    new_state = wolf_update(
        agent_state,
        experience.obs,
        experience.action,
        experience.reward,
        experience.next_obs,
        experience.done,
        config["LR"],
        config["DISCOUNT"],
        config["DELTA_WIN"],
        config["DELTA_LOSE"]
    )
    return new_state, {}
