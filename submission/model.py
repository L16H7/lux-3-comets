import functools
import jax
import math
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal
from typing import TypedDict


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
        
    # Normalize positions to [-1, 1]
    positions = positions / (max_size / 2) - 1
    
    # Generate frequency bands
    freq_bands = jnp.arange(embedding_dim // 4)
    freqs = 1.0 / (10000 ** (freq_bands / (embedding_dim // 4)))
    
    # Reshape for broadcasting
    x = positions[..., 0:1]  # (n_units, 1)
    y = positions[..., 1:2]  # (n_units, 1)
    freqs = freqs.reshape(1, -1)  # (1, embedding_dim//4)
    
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


import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import orthogonal
from typing import Tuple

class ResidualBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, self.kernel_size, self.strides, padding="SAME")(x)
        y = nn.leaky_relu(y)
        y = nn.Conv(self.features, self.kernel_size, self.strides, padding="SAME")(y)
        y += residual  # Adding the input x to the output of the convolution block
        return nn.leaky_relu(y)  # Apply activation after adding the residual


class ActorInput(TypedDict):
    positions: jax.Array
    states: jax.Array
    observations: jax.Array
    match_steps: jax.Array
    matches: jax.Array
    team_points: jax.Array
    opponent_points: jax.Array
    unit_move_cost: jax.Array
    unit_sap_cost: jax.Array
    unit_sap_range: jax.Array
    unit_sensor_range: jax.Array
    agent_ids: jax.Array
 
class Actor(nn.Module):
    n_actions: int = 6
    info_emb_dim: int = 32
    action_emb_dim: int = 16
    hidden_dim: int = 128
    position_emb_dim: int = 32
 
    @nn.compact
    def __call__(self, actor_input: ActorInput):
        state_encoder = nn.Sequential([
            nn.Conv(32, (3, 3), padding='SAME', kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            # ResidualBlock(32),
            nn.Conv(32, (3, 3), padding='SAME', kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(128),
            # nn.leaky_relu
        ])

        observation_encoder = nn.Sequential([
            nn.Conv(32, (2, 2), padding='SAME', kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            ResidualBlock(32),
            nn.Conv(32, (2, 2), padding='SAME', kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(256),
            nn.leaky_relu
        ])

        batch_size = actor_input['states'].shape[0]

        observation_embeddings = observation_encoder(
            actor_input['observations'].transpose((0, 2, 3, 1))
        )

        state_embeddings = state_encoder(
            actor_input['states'].transpose((0, 2, 3, 1))
        )

        position_embeddings = get_2d_positional_embeddings(
            actor_input['positions'],
            embedding_dim=32,
            max_size=24
        )

        info_input = jnp.stack([
            actor_input['team_points'],
            actor_input['opponent_points'],
            actor_input['match_steps'],
            actor_input['matches'],
        ], axis=-1)

        env_info_input = jnp.stack([
            actor_input['unit_move_cost'],
            actor_input['unit_sap_cost'],
            actor_input['unit_sap_range'],
            actor_input['unit_sensor_range'],
        ], axis=-1)

        info_embeddings = nn.Sequential([
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
        ])(info_input)

        env_info_embeddings = nn.Sequential([
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
        ])(env_info_input)

        actor = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.leaky_relu,
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.leaky_relu,
            ]
        )

        embeddings = jnp.concat([
            state_embeddings,
            info_embeddings,
            env_info_embeddings,
            position_embeddings,
            observation_embeddings,
        ], axis=-1)

        x = actor(embeddings)

        action_head = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))

        x_coordinate_head = nn.Dense(17, kernel_init=orthogonal(0.01))
        y_coordinate_head = nn.Dense(17, kernel_init=orthogonal(0.01))

        logits1 = action_head(x)
        logits2 = x_coordinate_head(x)
        logits3 = y_coordinate_head(x)

        return logits1, logits2, logits3


class CriticInput(TypedDict):
    states: jax.Array
    match_steps: jax.Array         # 4 phases each with 25 steps
    team_points: jax.Array
    opponent_points: jax.Array
 

class Critic(nn.Module):
    info_emb_dim: int = 32
    hidden_dim: int = 256
 
    @nn.compact
    def __call__(self, critic_input):
        seq_len, batch_size = critic_input['states'].shape[:2]

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
                lambda x: x.reshape((x.shape[0], -1)),
                nn.Dense(256),
                nn.leaky_relu,
            ]
        )
        state_embeddings = state_encoder(
            critic_input['states'].transpose((0, 2, 3, 1))
        )

        info_input = jnp.stack([
            critic_input['team_points'],
            critic_input['opponent_points'],
            critic_input['match_steps'],
            critic_input['matches'],
        ], axis=-1)


        info_embeddings = nn.Sequential([
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
        ])(info_input)

        embeddings = jnp.concat([
            state_embeddings,
            info_embeddings,
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

        values = critic(embeddings)
        return values
