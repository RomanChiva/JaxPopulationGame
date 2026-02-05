import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from algorithms.online_q.online_q import AgentState, create_agent_state, get_state_index

# Reuse AgentState and create function from Q-learning as they are identical for tabular methods
# We only need to redefine update and select_action (if we wanted different exploration, but eps-greedy is standard)

@partial(jax.jit, static_argnames=("epsilon",))
def select_action(agent_state: AgentState, obs: jnp.ndarray, key: jax.random.PRNGKey, epsilon: float) -> jnp.ndarray:
    """Selects an action using epsilon-greedy (same as Q-learning)."""
    state_idx = get_state_index(obs)
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
    next_action: jnp.ndarray, # SARSA needs next action
    done: jnp.ndarray,
    learning_rate: float,
    discount: float,
) -> AgentState:
    """Updates a single agent's Q-table using SARSA update rule."""
    current_state_idx = get_state_index(obs)
    next_state_idx = get_state_index(next_obs)

    # SARSA: Q(s,a) <- Q(s,a) + lr * (r + gamma * Q(s', a') - Q(s,a))
    # We use the actual next action taken by the policy
    next_q = agent_state.q_table[next_state_idx, next_action]
    
    target = reward + discount * next_q
    current_q = agent_state.q_table[current_state_idx, action]

    new_q = current_q + learning_rate * (target - current_q)
    updated_q_table = agent_state.q_table.at[current_state_idx, action].set(new_q)

    return agent_state.replace(q_table=updated_q_table)
