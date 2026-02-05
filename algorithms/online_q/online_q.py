# q_learning_logic.py
import jax
import jax.numpy as jnp
from flax import struct
from functools import partial


# 1. Define the state for a SINGLE agent.
@struct.dataclass
class AgentState:
    q_table: jnp.ndarray  # Shape: (num_total_states, num_actions)

def create_agent_state(key: jax.random.PRNGKey, num_obs_states: int, num_actions: int, config: dict) -> AgentState:
    """Initializes the state for a single agent with random Q-values."""
    initial_q_table = (
        jax.random.normal(key, shape=(num_obs_states, num_actions))
        * config["Q_INIT_STD"]
        + config["Q_INIT_MEAN"]
    )
    return AgentState(q_table=initial_q_table)

def get_state_index(obs: jnp.ndarray, obs_type: str = "standard") -> jnp.ndarray:
    
    
    size = obs.shape[0]
    
    # Branch based on size to avoid string overhead inside JIT if possible, 
    # but strictly speaking, explicit config is better. 
    # We will assume the caller handles the logic or we use size as proxy if safe.
    # Given the env changes:
    # Size 1 -> Opponent
    # Size 2 -> Standard
    # Size 3 -> Population
    
    def _get_opp_idx(_): # Size 1
        return obs[0].astype(jnp.int32)
        
    def _get_std_idx(_): # Size 2
        return (2 * obs[0] + obs[1]).astype(jnp.int32)
        
    def _get_pop_idx(_): # Size 3
        my_rep = obs[0].astype(jnp.int32)
        opp_rep = obs[1].astype(jnp.int32)
        avg_rep = obs[2]
        
        # Discretize avg_rep (0.0 to 1.0) into 10 bins (0..9)
        bin_idx = jnp.floor(avg_rep * 10).astype(jnp.int32)
        bin_idx = jnp.clip(bin_idx, 0, 9)
        
        base_idx = 2 * my_rep + opp_rep
        return base_idx * 10 + bin_idx

    # Dynamically dispatch based on shape size works well in JAX tracing
    idx = jax.lax.switch(size - 1, [_get_opp_idx, _get_std_idx, _get_pop_idx], None)
    return idx


@partial(jax.jit, static_argnames=("epsilon",))
def select_action(agent_state: AgentState, obs: jnp.ndarray, key: jax.random.PRNGKey, epsilon: float) -> jnp.ndarray:
    """Selects an action for a single agent using an epsilon-greedy policy."""
    state_idx = get_state_index(obs) # Logic checks shape of obs

    explore_key, action_key = jax.random.split(key)

    explore = jax.random.uniform(explore_key) < epsilon
    random_action = jax.random.randint(action_key, shape=(), minval=0, maxval=2)
    best_action = jnp.argmax(agent_state.q_table[state_idx])

    return jnp.where(explore, random_action, best_action)

@partial(jax.jit, static_argnames=("learning_rate", "discount"))
def update(
    agent_state: AgentState,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    next_obs: jnp.ndarray,
    done: jnp.ndarray,
    learning_rate: float,
    discount: float,
) -> AgentState:
    """Updates a single agent's Q-table based on a transition."""
    current_state_idx = get_state_index(obs)
    next_state_idx = get_state_index(next_obs)

    best_next_q = jnp.max(agent_state.q_table[next_state_idx])
    target = reward + discount * best_next_q # Ignoring done for infinite horizon/stateless games sometimes, but let's keep it simple
    current_q = agent_state.q_table[current_state_idx, action]

    new_q = current_q + learning_rate * (target - current_q)
    updated_q_table = agent_state.q_table.at[current_state_idx, action].set(new_q)

    return agent_state.replace(q_table=updated_q_table)
