import jax
import jax.numpy as jnp
from flax import struct
from functools import partial
from algorithms.online_q.online_q import get_state_index

# WolF-PHC (Win or Learn Fast - Policy Hill Climbing)
# Maintains Q(s,a) and Pi(s,a).
# Also maintains avg_Pi(s,a) (average policy over time) to determine "Win" or "Lose".

@struct.dataclass
class AgentState:
    q_table: jnp.ndarray      # Q(s,a)
    policy: jnp.ndarray       # \pi(s,a)
    avg_policy: jnp.ndarray   # \bar{\pi}(s,a)
    policy_count: jnp.ndarray # C(s), count of visits to state s to update avg_policy

def create_agent_state(key: jax.random.PRNGKey, num_obs_states: int, num_actions: int, config: dict) -> AgentState:
    # Init Q
    initial_q = (jax.random.normal(key, (num_obs_states, num_actions)) 
                 * config["Q_INIT_STD"] + config["Q_INIT_MEAN"])
    
    # Init Policy (Uniform)
    initial_policy = jnp.ones((num_obs_states, num_actions)) / num_actions
    
    # Init Avg Policy (Uniform)
    initial_avg_policy = jnp.ones((num_obs_states, num_actions)) / num_actions
    
    # Init Count
    initial_count = jnp.zeros((num_obs_states,))
    
    return AgentState(
        q_table=initial_q, 
        policy=initial_policy, 
        avg_policy=initial_avg_policy,
        policy_count=initial_count
    )

@partial(jax.jit, static_argnames=())
def select_action(agent_state: AgentState, obs: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
    """Selects action based on the current policy \\pi(s)."""
    state_idx = get_state_index(obs)
    probs = agent_state.policy[state_idx]
    return jax.random.categorical(key, jnp.log(probs + 1e-8))

@partial(jax.jit, static_argnames=("learning_rate", "discount", "delta_win", "delta_lose"))
def update(
    agent_state: AgentState,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    next_obs: jnp.ndarray,
    done: jnp.ndarray,
    learning_rate: float,
    discount: float,
    delta_win: float,
    delta_lose: float,
) -> AgentState:
    
    s = get_state_index(obs)
    next_s = get_state_index(next_obs)
    
    # 1. Update Q-value (Standard Q-Learning)
    best_next_q = jnp.max(agent_state.q_table[next_s])
    target = reward + discount * best_next_q
    cur_q = agent_state.q_table[s, action]
    new_q = cur_q + learning_rate * (target - cur_q)
    new_q_table = agent_state.q_table.at[s, action].set(new_q)
    
    # 2. Update Average Policy
    # \bar{\pi}(s,a) <- \bar{\pi}(s,a) + (1/C(s)) * (\pi(s,a) - \bar{\pi}(s,a))
    count = agent_state.policy_count[s] + 1
    new_count_arr = agent_state.policy_count.at[s].set(count)
    
    current_pi = agent_state.policy[s]
    current_avg_pi = agent_state.avg_policy[s]
    new_avg_pi_s = current_avg_pi + (1.0 / count) * (current_pi - current_avg_pi)
    new_avg_policy = agent_state.avg_policy.at[s].set(new_avg_pi_s)
    
    # 3. Determine Delta (Win or Lose?)
    # Sum_{a} \pi(s,a) * Q(s,a)
    expected_val_pi = jnp.dot(current_pi, new_q_table[s])
    # Sum_{a} \bar{\pi}(s,a) * Q(s,a)
    expected_val_avg = jnp.dot(new_avg_pi_s, new_q_table[s])
    
    # Note: Use values from CURRENT step (with updated Q)
    
    is_winning = expected_val_pi >= expected_val_avg
    delta = jnp.where(is_winning, delta_win, delta_lose)
    
    # 4. Update Policy (Hill Climbing)
    # Move towards optimal action (argmax Q)
    # We need argmax Q(s,a). If ties, we should technically distribute, but jnp.argmax takes first.
    # For robust WolF, let's assume unique max or just take one.
    best_a = jnp.argmax(new_q_table[s])
    
    # We want to increase prob of best_a by delta, and decrease others.
    # But constraints: sum = 1, probs >= 0.
    
    # Vectorized update for policy[s]:
    # p_new(a) = p(a) + delta if a == best_a
    # p_new(a) = p(a) - delta/(num_actions-1) if a != best_a
    # BUT we must clip to valid range [0, 1] and ensure sum is 1.
    
    state_probs = current_pi
    
    # Calculate how much we CAN decrease 'other' actions
    # We can at most take min(current_prob, delta / (num_actions - 1)) from each other action
    num_actions = state_probs.shape[0]
    
    # Simple logic:
    # 1. Identify suboptimal actions
    suboptimal_mask = jnp.arange(num_actions) != best_a
    
    # 2. Reduction budget
    # We want to reduce each suboptimal action by delta / (num_actions - 1)
    reduce_step = delta / (num_actions - 1)
    
    # 3. Apply reduction, clamping at 0
    # Available to take: state_probs * suboptimal_mask
    reduction = jnp.minimum(state_probs, reduce_step) * suboptimal_mask
    
    # 4. Total reduced mass
    total_reduction = jnp.sum(reduction)
    
    # 5. Apply changes
    new_probs = state_probs - reduction
    new_probs = new_probs.at[best_a].add(total_reduction)
    
    new_policy = agent_state.policy.at[s].set(new_probs)
    
    return agent_state.replace(
        q_table=new_q_table,
        policy=new_policy,
        avg_policy=new_avg_policy,
        policy_count=new_count_arr
    )
