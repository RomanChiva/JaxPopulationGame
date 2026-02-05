
import jax
import jax.numpy as jnp
from algorithms.online_sarsa.online_sarsa import create_agent_state, select_action as sarsa_select, update as sarsa_update

def init(key, config, obs_shape, num_actions):
    obs_dim = obs_shape[0]
    if obs_dim == 1: num_states = 2 
    elif obs_dim == 2: num_states = 4
    elif obs_dim == 3: num_states = 40
    else: num_states = 4
    return create_agent_state(key, num_states, num_actions, config)

def select_action(agent_state, obs, key, config):
    action = sarsa_select(agent_state, obs, key, config["EPSILON"])
    return action, ()

def update(agent_state, experience, key, config):
    # SARSA needs (S, A, R, S', A')
    # We have S, A, R, S'
    # We need to generate A' using the CURRENT policy.
    # Note: select_action uses RNG. We need RNG.
    
    # Generate A'
    next_action = sarsa_select(
        agent_state,
        experience.next_obs,
        key, # Reuse the key passed to update? Shared loop splits unique key for update.
        config["EPSILON"]
    )
    
    new_state = sarsa_update(
        agent_state,
        experience.obs,
        experience.action,
        experience.reward,
        experience.next_obs,
        next_action,
        experience.done,
        config["LR"],
        config["DISCOUNT"]
    )
    
    return new_state, {}
