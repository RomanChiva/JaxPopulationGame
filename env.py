import jax
import jax.numpy as jnp
import chex
from flax import struct
from functools import partial

# --- MODIFIED EnvState ---
# Added a `timestep` counter.
@struct.dataclass
class EnvState:
    reputations: jnp.ndarray
    timestep: jnp.ndarray
    matchups: jnp.ndarray 


class PopulationEnv:

    # --- MODIFIED __init__ ---
    # Added `max_steps_per_episode` and `obs_type`.
    def __init__(self, num_agents: int, norm_string: str, b: float, c: float, max_steps_per_episode=100000000, obs_type: str = "standard"):

        if num_agents % 2 != 0:
            raise ValueError("`num_agents` must be an even number for pairwise matching.")

        self.num_agents = num_agents
        self.num_actions = 2
        
        self.obs_type = obs_type
        if self.obs_type == "opponent":
            self.observation_shape = (1,)
        elif self.obs_type == "standard":
            self.observation_shape = (2,)
        elif self.obs_type == "population":
            self.observation_shape = (3,)
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")

        self.action_shape = ()
        # --- Store max steps ---
        self.max_steps_per_episode = max_steps_per_episode

        if len(norm_string) != 4 or not all(char in '01' for char in norm_string):
            raise ValueError("`norm_string` must be a 4-bit string (e.g., '0110').")
        self.norm_table = jnp.array([int(bit) for bit in norm_string], dtype=jnp.int32)
        # jax.debug.print("Using norm table: {}", self.norm_table)
        jax.debug.print("Initialized PopulationEnv with norm table: {}", self.norm_table)
        

        R = b - c
        S = -c
        T = b
        P = 0.0
        self._payoff_matrix = jnp.array([[P, T], [S, R]])

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> tuple[jnp.ndarray, EnvState]:
        """
        Resets the environment. Reputations are randomized and timestep is set to 0.
        """
        key, reps_init_key = jax.random.split(key)

        initial_reputations = jax.random.randint(reps_init_key, (self.num_agents,), 0, 2).astype(jnp.int32)
       
         # Create initial matchups
        key, matchup_key = jax.random.split(key)
        initial_matchups = jax.random.permutation(matchup_key, self.num_agents).reshape(-1, 2)

        # --- FIX: Store matchups in the initial state ---
        state = EnvState(
            reputations=initial_reputations,
            timestep=jnp.array(0, dtype=jnp.int32),
            matchups=initial_matchups
        )

        obs = self._get_obs_from_state(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self, state: EnvState, actions: jnp.ndarray, key: chex.PRNGKey
    ) -> tuple[jnp.ndarray, EnvState, jnp.ndarray, jnp.ndarray, dict]:
        """
        Performs one step, increments the timer, and checks for episode termination.
        """
        # 1. Get matchups directly from the input state
        p1_indices, p2_indices = state.matchups[:, 0], state.matchups[:, 1]

        # 2. Get actions and reputations for each pair
        p1_actions = actions[p1_indices]
        p2_actions = actions[p2_indices]
        p1_reps = state.reputations[p1_indices]
        p2_reps = state.reputations[p2_indices]

        # 3. Calculate rewards
        p1_rewards = self._payoff_matrix[p1_actions, p2_actions]
        p2_rewards = self._payoff_matrix[p2_actions, p1_actions]
        rewards = jnp.zeros(self.num_agents).at[p1_indices].set(p1_rewards)
        rewards = rewards.at[p2_indices].set(p2_rewards)

        
        p1_norm_indices = 2 * p2_reps + p1_actions
        p2_norm_indices = 2 * p1_reps + p2_actions
        new_p1_reps = self.norm_table[p1_norm_indices]
        new_p2_reps = self.norm_table[p2_norm_indices]
        next_reputations = jnp.zeros_like(state.reputations)
        next_reputations = next_reputations.at[p1_indices].set(new_p1_reps)
        next_reputations = next_reputations.at[p2_indices].set(new_p2_reps)

         # 5. Create new matchups for the *next* state
        key, next_matchup_key = jax.random.split(key)
        next_matchups = jax.random.permutation(next_matchup_key, self.num_agents).reshape(-1, 2)

        # 6. Create the next state, including the new matchups
        next_state = state.replace(
            reputations=next_reputations,
            timestep=state.timestep + 1,
            matchups=next_matchups  # <-- Store the new matchups
        )


        # 5b. Check if the episode should end
        done = next_state.timestep >= self.max_steps_per_episode
        dones = jnp.full((self.num_agents,), done, dtype=jnp.bool_)
        # --- END OF MODIFICATIONS ---


        info = {
            "avg_reward": jnp.mean(rewards),
            "cooperation_rate": jnp.mean(actions == 1),
            "good_reputation_rate": jnp.mean(next_reputations),
        }

        next_obs = self._get_obs_from_state(next_state)

        return next_obs, next_state, rewards, dones, info

    def _get_obs_from_state(self, state: EnvState) -> jnp.ndarray:
        # Get matchups from the state object
        p1_indices, p2_indices = state.matchups[:, 0], state.matchups[:, 1]
        p1_reps = state.reputations[p1_indices]
        p2_reps = state.reputations[p2_indices]
        
        obs_dtype = jnp.float32 # Use float to accommodate averages if needed, though reps are ints
        
        if self.obs_type == "opponent":
            # Obs: [opp_rep]
            p1_obs = p2_reps.reshape(-1, 1).astype(obs_dtype)
            p2_obs = p1_reps.reshape(-1, 1).astype(obs_dtype)
            obs_width = 1
        elif self.obs_type == "standard":
            # Obs: [my_rep, opp_rep]
            p1_obs = jnp.stack([p1_reps, p2_reps], axis=1).astype(obs_dtype)
            p2_obs = jnp.stack([p2_reps, p1_reps], axis=1).astype(obs_dtype)
            obs_width = 2
        elif self.obs_type == "population":
            # Obs: [my_rep, opp_rep, avg_rep_in_population]
            mean_rep = jnp.mean(state.reputations).astype(obs_dtype)
            # Create a column of mean reps matching the batch size of p1/p2 (which is num_agents/2)
            mean_col = jnp.full((self.num_agents // 2,), mean_rep)
            
            p1_obs = jnp.stack([p1_reps, p2_reps, mean_col], axis=1).astype(obs_dtype)
            p2_obs = jnp.stack([p2_reps, p1_reps, mean_col], axis=1).astype(obs_dtype)
            obs_width = 3

        obs = jnp.zeros((self.num_agents, obs_width), dtype=obs_dtype)
        obs = obs.at[p1_indices].set(p1_obs)
        obs = obs.at[p2_indices].set(p2_obs)
        return obs

# Example of how to use the environment (for testing)
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    num_agents = 10
    episode_length = 500  # Example episode length

    env = PopulationEnv(
        num_agents=num_agents,
        norm_string="0110",
        b=5.0,
        c=1.0,
        max_steps_per_episode=episode_length
    )

    key, reset_key = jax.random.split(key)
    obs, state = env.reset(reset_key)
    print(f"Initial state: {state}")

    # Simulate a few steps
    for i in range(5):
        key, action_key, step_key = jax.random.split(key, 3)
        actions = jax.random.randint(action_key, shape=(num_agents,), minval=0, maxval=2)
        obs, state, rewards, dones, info = env.step(state, actions, step_key)
        print(f"Step {state.timestep}, Done: {dones[0]}")

    # Simulate jumping to the end of an episode
    print("\nSimulating final step of episode...")
    final_step_state = state.replace(timestep=jnp.array(episode_length - 1))
    key, action_key, step_key = jax.random.split(key, 3)
    actions = jax.random.randint(action_key, shape=(num_agents,), minval=0, maxval=2)
    obs, state, rewards, dones, info = env.step(final_step_state, actions, step_key)
    print(f"Step {state.timestep}, Done: {dones[0]}")
