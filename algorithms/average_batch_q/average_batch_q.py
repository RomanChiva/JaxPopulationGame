import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from algorithms.online_q.online_q import AgentState, create_agent_state, get_state_index

# For Batch Q, the state is the same (Q-table).
# However, the update function will aggregate gradients/values differently.
# But since our training loop in main.py is likely vmapped over agents, 
# "Average Batch Q" in this context likely means we average the *update* over a batch of experiences
# collected before applying to the Q-table. 
# Or, if this is "Fitted Q Iteration" logic. 
# Given the user says "Average Batch Q", I will interpret it as accumulating gradients over a batch of time steps 
# and then applying them. Or better: it's just Q-learning but we implement a `batch_update` function
# that takes a batch of transitions.

@struct.dataclass
class BatchTransition:
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray

@partial(jax.jit, static_argnames=("epsilon",))
def select_action(agent_state: AgentState, obs: jnp.ndarray, key: jax.random.PRNGKey, epsilon: float) -> jnp.ndarray:
    return jax.jit(partial(from_online_q_select_action, epsilon=epsilon))(agent_state, obs, key)

# Import logic from online_q to avoid duplication if possible, or just copy-paste for safety
from algorithms.online_q.online_q import select_action as from_online_q_select_action

@partial(jax.jit, static_argnames=("learning_rate", "discount"))
def batch_update(
    agent_state: AgentState,
    batch: BatchTransition, # Shape: (Batch_Size, ...)
    learning_rate: float,
    discount: float,
) -> AgentState:
    """
    Updates Q-table based on a batch of transitions. 
    It calculates the mean target for each state-action pair visited in the batch
    and applies a single update.
    """
    
    # 1. Calculate targets for the whole batch
    state_indices = jax.vmap(get_state_index)(batch.obs)
    next_state_indices = jax.vmap(get_state_index)(batch.next_obs)
    
    best_next_qs = jnp.max(agent_state.q_table[next_state_indices], axis=1)
    targets = batch.reward + discount * best_next_qs
    
    # 2. We need to aggregate updates for the same (s, a) pair.
    # Q_new(s,a) = Q_old(s,a) + lr * mean(target - Q_old(s,a)) over occurrences
    
    # Current Q values for the actions taken
    current_qs = agent_state.q_table[state_indices, batch.action]
    td_errors = targets - current_qs
    
    # We want to scatter_add these td_errors to the appropriate table entries
    # and then divide by counts.
    
    # Flat index for (s, a)
    flat_indices = state_indices * agent_state.q_table.shape[1] + batch.action
    
    # Initialize accumulators
    total_states = agent_state.q_table.size
    grad_acc = jnp.zeros(total_states)
    count_acc = jnp.zeros(total_states)
    
    grad_acc = grad_acc.at[flat_indices].add(td_errors)
    count_acc = count_acc.at[flat_indices].add(1.0)
    
    # Avoid division by zero
    safe_counts = jnp.maximum(count_acc, 1.0)
    avg_td_errors = grad_acc / safe_counts
    
    # Apply update only where counts > 0
    mask = count_acc > 0
    
    flat_q = agent_state.q_table.ravel()
    new_flat_q = flat_q + mask * learning_rate * avg_td_errors
    
    updated_q_table = new_flat_q.reshape(agent_state.q_table.shape)
    
    return agent_state.replace(q_table=updated_q_table)

