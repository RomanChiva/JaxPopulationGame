# ppo_agent.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax

# --- Same ActorCritic Network from your script ---
class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"


    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        # N Layers
        size_ = 32

        actor_mean = nn.Dense(size_, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(size_, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(size_, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(size_, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)

# --- Same Transition structure ---
class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray

# --- Agent Initialization Function ---
def create_agent_state(key, config, obs_shape, action_dim):
    """Initializes the TrainState for a single agent."""
    network = ActorCritic(action_dim, activation=config["ACTIVATION"])
    key, net_key = jax.random.split(key)
    init_x = jnp.zeros(obs_shape)
    network_params = network.init(net_key, init_x)
    
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    return train_state

# --- GAE Calculation (extracted for modularity) ---
def calculate_gae(traj_batch: Transition, last_val: jnp.ndarray, config: dict):
    """Calculates the GAE for a single agent's trajectory."""
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + config["GAMMA"] * next_value * (1 - done) - value
        gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
    )
    return advantages, advantages + traj_batch.value

# --- The Pure PPO Update Function ---
def ppo_update(train_state: TrainState, traj_batch: Transition, advantages: jnp.ndarray, targets: jnp.ndarray, rng: jax.random.PRNGKey, config: dict):
    """
    Performs the PPO update for a single agent. The gradients are calculated
    and applied internally in this function.
    """
    batch_size = config["NUM_STEPS"]
    minibatch_size = batch_size // config["NUM_MINIBATCHES"]
    
    def _update_epoch(carry, unused):
        train_state, rng = carry
        
        # Shuffle data
        rng, perm_key = jax.random.split(rng)
        permutation = jax.random.permutation(perm_key, batch_size)
        batch = (traj_batch, advantages, targets)
        shuffled_batch = jax.tree.map(lambda x: x[permutation], batch)
        
        # Reshape to minibatches
        minibatches = jax.tree.map(
            lambda x: x.reshape((config["NUM_MINIBATCHES"], minibatch_size) + x.shape[1:]),
            shuffled_batch,
        )

        def _loss_fn(params, traj, gae, targets):
            pi, value = train_state.apply_fn(params, traj.obs)
            log_prob = pi.log_prob(traj.action)

            # Value loss
            value_loss = jnp.square(value - targets).mean()

            # Actor loss
            ratio = jnp.exp(log_prob - traj.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            loss_actor1 = ratio * gae
            loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
            
            # Entropy bonus
            entropy = pi.entropy().mean()

            total_loss = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
            return total_loss, (value_loss, loss_actor, entropy)
        
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)

        def _update_minibatch(ts, minibatch):
            (loss, (v_loss, a_loss, ent)), grads = grad_fn(ts.params, *minibatch)
            new_ts = ts.apply_gradients(grads=grads)
            return new_ts, (loss, v_loss, a_loss, ent)


        final_train_state, metrics = jax.lax.scan(_update_minibatch, train_state, minibatches)
        return (final_train_state, rng), metrics

    (final_train_state, _), metrics = jax.lax.scan(
        _update_epoch, (train_state, rng), None, config["UPDATE_EPOCHS"]
    )
    
    # metrics is shaped (UPDATE_EPOCHS, NUM_MINIBATCHES, 4), take mean over all
    avg_metrics = jax.tree.map(lambda x: x.mean(), metrics)
    
    return final_train_state, {
        "total_loss": avg_metrics[0],
        "value_loss": avg_metrics[1],
        "actor_loss": avg_metrics[2],
        "entropy": avg_metrics[3],
    }
