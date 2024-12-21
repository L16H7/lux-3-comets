import functools
import jax
import math
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal
from typing import TypedDict


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        new_rnn_state, y = nn.GRUCell(features=x.shape[1])(rnn_state, x)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


def get_2d_positional_embeddings(positions, embedding_dim=32, max_size=24):
    """
    Generate positional embeddings for 2D coordinates.
    
    Args:
        positions: Array of shape (n_envs, n_units, 2) containing x,y coordinates
        embedding_dim: Dimension of the output embeddings (must be divisible by 4)
        max_size: Maximum size of the grid (used for scaling)
    
    Returns:
        Array of shape (n_envs, n_units, embedding_dim) containing positional embeddings
    """
    if embedding_dim % 4 != 0:
        raise ValueError("embedding_dim must be divisible by 4")
        
    n_envs, n_units, _ = positions.shape
    
    # Normalize positions to [-1, 1]
    positions = positions / (max_size / 2) - 1
    
    # Generate frequency bands
    freq_bands = jnp.arange(embedding_dim // 4)
    freqs = 1.0 / (10000 ** (freq_bands / (embedding_dim // 4)))
    
    # Reshape for broadcasting
    x = positions[..., 0:1]  # (n_envs, n_units, 1)
    y = positions[..., 1:2]  # (n_envs, n_units, 1)
    freqs = freqs.reshape(1, 1, -1)  # (1, 1, embedding_dim//4)
    
    # Calculate embeddings for x and y separately
    x_sines = jnp.sin(x * freqs * jnp.pi)
    x_cosines = jnp.cos(x * freqs * jnp.pi)
    y_sines = jnp.sin(y * freqs * jnp.pi)
    y_cosines = jnp.cos(y * freqs * jnp.pi)
    
    # Concatenate all components
    embeddings = jnp.concatenate(
        [x_sines, x_cosines, y_sines, y_cosines],
        axis=-1
    )
    
    return embeddings


class ActorInput(TypedDict):
    positions: jax.Array
    observations: jax.Array
    match_phases: jax.Array
    matches: jax.Array
    team_points: jax.Array
    opponent_points: jax.Array
    prev_points: jax.Array
    prev_actions: jax.Array
    unit_move_cost: jax.Array
    unit_sap_cost: jax.Array
    unit_sap_range: jax.Array
    unit_sensor_range: jax.Array
 
class Actor(nn.Module):
    n_actions: int = 6
    info_emb_dim: int = 32
    action_emb_dim: int = 16
    hidden_dim: int = 128
    position_emb_dim: int = 32
 
    @nn.compact
    def __call__(self, hstate: jax.Array, actor_input: ActorInput):
        observation_encoder = nn.Sequential(
            [
                nn.Conv(
                    32,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.LayerNorm(),
                nn.Conv(
                    32,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.LayerNorm(),
                nn.Conv(
                    32,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.LayerNorm(),
                nn.Conv(
                    32,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.LayerNorm(),
                lambda x: x.reshape((x.shape[0], x.shape[1], -1)),
                nn.Dense(256),
                nn.leaky_relu,
                nn.LayerNorm(),
            ]
        )

        observation_embeddings = observation_encoder(actor_input['observations'])

        position_embeddings = get_2d_positional_embeddings(
            actor_input['positions'],
            embedding_dim=32,  # Must be divisible by 4
            max_size=24
        )

        prev_action_embeddings = nn.Embed(
            self.n_actions,
            self.action_emb_dim
        )(actor_input['prev_actions'])

        info_input = jnp.concat([
            actor_input['match_phases'],
            actor_input['matches'],
            actor_input['prev_points'],
            actor_input['team_points'],
            actor_input['opponent_points'],
            actor_input['unit_move_cost'],
            actor_input['unit_sap_cost'],
            actor_input['unit_sap_range'],
            actor_input['unit_sensor_range'],
        ], axis=-1)

        info_embeddings = nn.Sequential([
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            nn.LayerNorm()
        ])(info_input)

        embeddings = jnp.concat([
            position_embeddings,
            observation_embeddings,
            prev_action_embeddings,
            info_embeddings,
        ], axis=-1)

        actor = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.leaky_relu,
            ]
        )
        # Normalize before RNN
        embeddings = nn.LayerNorm()(embeddings)

        hstate, out = ScannedRNN()(hstate, embeddings)
        x = actor(out)
        logits1 = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))(x)
        logits2 = nn.Dense(17, kernel_init=orthogonal(0.01))(x)
        logits3 = nn.Dense(17, kernel_init=orthogonal(0.01))(x)
       
        return [logits1, logits2, logits3], hstate


class CriticInput(TypedDict):
    states: jax.Array
    match_phases: jax.Array         # 4 phases each with 25 steps
    team_points: jax.Array
    opponent_points: jax.Array
 

class Critic(nn.Module):
    match_phase_emb_dim: int = 16
    point_info_emb_dim: int = 16
    hidden_dim: int = 256
 
    @nn.compact
    def __call__(self, hstate: jax.Array, critic_input):
        state_encoder = nn.Sequential(
            [
                nn.Conv(
                    128,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    128,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    128,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    128,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                lambda x: x.reshape((x.shape[0], x.shape[1], -1)),
                nn.Dense(256),
                nn.leaky_relu,
            ]
        )

        state_embeddings = state_encoder(critic_input['states'])

        match_phase_embeddings = nn.Embed(4, self.match_phase_emb_dim)(critic_input['match_phases'])

        point_info_embeddings = nn.Dense(self.point_info_emb_dim)(
            jnp.concat([
                critic_input['team_points'],
                critic_input['opponent_points'],
            ], axis=-1)
        )

        embeddings = jnp.concat([
            state_embeddings,
            match_phase_embeddings,
            point_info_embeddings,
        ], axis=-1)

        critic = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.leaky_relu,
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        hstate, out = ScannedRNN()(hstate, embeddings)
        values = critic(out)
        return values, hstate
