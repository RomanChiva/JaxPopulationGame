
import sys
import os
sys.path.append(os.getcwd())

import jax
import jax.numpy as jnp
import numpy as np
from env import PopulationEnv

def test_obs_types():
    norm = "0110"
    num_agents = 4
    
    # Test 1: Opponent (Size 1)
    print("Testing 'opponent' obs type...")
    env = PopulationEnv(num_agents, norm, 5.0, 1.0, obs_type="opponent")
    key = jax.random.PRNGKey(0)
    obs, state = env.reset(key)
    print(f"Obs shape: {obs.shape}")
    assert obs.shape == (num_agents, 1)
    
    # Test 2: Standard (Size 2)
    print("Testing 'standard' obs type...")
    env = PopulationEnv(num_agents, norm, 5.0, 1.0, obs_type="standard")
    obs, state = env.reset(key)
    print(f"Obs shape: {obs.shape}")
    assert obs.shape == (num_agents, 2)
    
    # Test 3: Population (Size 3)
    print("Testing 'population' obs type...")
    env = PopulationEnv(num_agents, norm, 5.0, 1.0, obs_type="population")
    obs, state = env.reset(key)
    print(f"Obs shape: {obs.shape}")
    assert obs.shape == (num_agents, 3)
    
    # Verify average rep value
    avg_rep_idx = 2
    actual_mean = jnp.mean(state.reputations)
    obs_mean = obs[0, avg_rep_idx]
    print(f"Actual mean: {actual_mean}, Obs mean: {obs_mean}")
    assert jnp.isclose(actual_mean, obs_mean)

    print("All environment observation tests passed!")

if __name__ == "__main__":
    test_obs_types()
