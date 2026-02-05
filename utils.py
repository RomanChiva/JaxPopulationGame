
# utils.py
import jax
import jax.numpy as jnp
from typing import Any
from flax.training.train_state import TrainState


def calculate_cooperation_probabilities(
    population_train_states: Any,
    actor_critic_class: Any,
    num_actions: int = 2
) -> jnp.ndarray:
    """
    Calculate the probability of cooperation for each agent at each possible state.
    
    Args:
        population_train_states: A pytree of TrainState objects for all agents
        actor_critic_class: The ActorCritic class (not an instance)
        num_actions: Number of possible actions (default 2: defect=0, cooperate=1)
    
    Returns:
        cooperation_probs: Array of shape (NUM_AGENTS, 4) where each row contains
                          the probability of cooperation for states:
                          [0,0], [0,1], [1,0], [1,1]
                          (own_reputation, partner_reputation)
    """
    # Define all 4 possible states
    # State format: [own_reputation, partner_reputation]
    all_states = jnp.array([
        [0, 0],  # Both bad reputation
        [0, 1],  # Self bad, partner good
        [1, 0],  # Self good, partner bad
        [1, 1],  # Both good reputation
    ], dtype=jnp.float32)
    
    def get_agent_coop_probs(train_state: TrainState) -> jnp.ndarray:
        """Get cooperation probabilities for a single agent across all states."""
        # Forward pass for all 4 states at once
        pi, _ = actor_critic_class(num_actions).apply(train_state.params, all_states)
        
        # Get the probabilities for each action
        # pi.probs has shape (4, num_actions) where dim 0 is the state, dim 1 is the action
        action_probs = pi.probs
        
        # Extract cooperation probability (action=1)
        coop_probs = action_probs[:, 1]  # Shape: (4,)
        
        return coop_probs
    
    # Vmap over all agents in the population
    vmapped_get_coop_probs = jax.vmap(get_agent_coop_probs)
    
    # Calculate cooperation probabilities for all agents
    # Result shape: (NUM_AGENTS, 4)
    cooperation_probs = vmapped_get_coop_probs(population_train_states)
    
    return cooperation_probs




import matplotlib.pyplot as plt
import jax.numpy as jnp
import numpy as np

def plot_cooperation_evolution(avg_coop_probs: jnp.ndarray, alpha: float = 0.3, 
                               linewidth: float = 1.0, figsize: tuple = (14, 10)):
    """
    Plot the evolution of cooperation probabilities for each agent in a 2x2 grid.
    
    Args:
        avg_coop_probs: Array of shape (NUM_UPDATES, NUM_AGENTS, 4) containing
                       cooperation probabilities for each agent at each update step
                       and for each of the 4 states.
        alpha: Transparency for individual agent lines (default 0.3)
        linewidth: Width of the lines (default 1.0)
        figsize: Figure size as (width, height) tuple (default (14, 10))
    """
    # Convert to numpy for plotting
    coop_probs = np.array(avg_coop_probs)
    
    # Shape should be (NUM_UPDATES, NUM_AGENTS, 4)
    num_updates, num_agents, num_states = coop_probs.shape
    
    # Define state labels
    state_labels = [
        "State [0,0]: Bad-Bad",
        "State [0,1]: Bad-Good", 
        "State [1,0]: Good-Bad",
        "State [1,1]: Good-Good"
    ]
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Color map for agents
    colors = plt.cm.viridis(np.linspace(0, 1, num_agents))
    
    # Plot each state in a separate subplot
    for state_idx in range(4):
        ax = axes[state_idx]
        
        # Plot each agent's trajectory
        for agent_idx in range(num_agents):
            ax.plot(
                coop_probs[:, agent_idx, state_idx],
                color=colors[agent_idx],
                alpha=alpha,
                linewidth=linewidth
            )
        
        # Plot population mean as a thick black line
        mean_coop = coop_probs[:, :, state_idx].mean(axis=1)
        ax.plot(
            mean_coop,
            color='black',
            linewidth=3.0,
            label='Population Mean',
            linestyle='--'
        )
        
        # Formatting
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Cooperation Probability', fontsize=12)
        ax.set_title(state_labels[state_idx], fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=10)
        
        # Add text box with final statistics
        final_mean = mean_coop[-1]
        final_std = coop_probs[-1, :, state_idx].std()
        textstr = f'Final: μ={final_mean:.3f}, σ={final_std:.3f}'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Evolution of Cooperation Probabilities Across All Agents', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()


def plot_cooperation_evolution_with_variance(avg_coop_probs: jnp.ndarray, 
                                            figsize: tuple = (14, 10)):
    """
    Plot the evolution of cooperation probabilities showing mean and variance bands.
    
    Args:
        avg_coop_probs: Array of shape (NUM_UPDATES, NUM_AGENTS, 4) containing
                       cooperation probabilities for each agent at each update step
                       and for each of the 4 states.
        figsize: Figure size as (width, height) tuple (default (14, 10))
    """
    # Convert to numpy for plotting
    coop_probs = np.array(avg_coop_probs)
    
    # Shape should be (NUM_UPDATES, NUM_AGENTS, 4)
    num_updates, num_agents, num_states = coop_probs.shape
    
    # Define state labels
    state_labels = [
        "State [0,0]: Bad-Bad",
        "State [0,1]: Bad-Good", 
        "State [1,0]: Good-Bad",
        "State [1,1]: Good-Good"
    ]
    
    # Create 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Plot each state in a separate subplot
    for state_idx in range(4):
        ax = axes[state_idx]
        
        # Calculate statistics across agents
        mean_coop = coop_probs[:, :, state_idx].mean(axis=1)
        std_coop = coop_probs[:, :, state_idx].std(axis=1)
        min_coop = coop_probs[:, :, state_idx].min(axis=1)
        max_coop = coop_probs[:, :, state_idx].max(axis=1)
        
        x = np.arange(num_updates)
        
        # Plot individual agents with low alpha
        for agent_idx in range(num_agents):
            ax.plot(
                coop_probs[:, agent_idx, state_idx],
                color='gray',
                alpha=0.15,
                linewidth=0.5
            )
        
        # Plot mean
        ax.plot(x, mean_coop, color='blue', linewidth=3.0, 
                label='Mean', zorder=10)
        
        # Plot standard deviation band
        ax.fill_between(x, mean_coop - std_coop, mean_coop + std_coop,
                        color='blue', alpha=0.2, label='±1 Std Dev')
        
        # Plot min-max range
        ax.fill_between(x, min_coop, max_coop,
                        color='gray', alpha=0.1, label='Min-Max Range')
        
        # Formatting
        ax.set_xlabel('Update Step', fontsize=12)
        ax.set_ylabel('Cooperation Probability', fontsize=12)
        ax.set_title(state_labels[state_idx], fontsize=14, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=9, loc='best')
        
        # Add text box with final statistics
        final_mean = mean_coop[-1]
        final_std = std_coop[-1]
        final_min = min_coop[-1]
        final_max = max_coop[-1]
        textstr = f'Final:\nμ={final_mean:.3f}\nσ={final_std:.3f}\nrange=[{final_min:.3f}, {final_max:.3f}]'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.suptitle(f'Evolution of Cooperation Probabilities (N={num_agents} agents)', 
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()