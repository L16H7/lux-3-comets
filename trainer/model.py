import jax
import math
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import orthogonal
from typing import TypedDict, Tuple


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


def sinusoidal_positional_encoding(seq_len, dim):
    """Create static sinusoidal positional encoding."""
    position = jnp.arange(0, seq_len)[:, jnp.newaxis]
    div_term = jnp.exp(jnp.arange(0, dim, 2) * -(jnp.log(10000.0) / dim))
    
    positional_embeddings = jnp.zeros((seq_len, dim))
    positional_embeddings = positional_embeddings.at[:, 0::2].set(jnp.sin(position * div_term))
    positional_embeddings = positional_embeddings.at[:, 1::2].set(jnp.cos(position * div_term))
    
    return positional_embeddings


class ActorInput(TypedDict):
    positions: jax.Array
    states: jax.Array
    match_steps: jax.Array
    matches: jax.Array
    team_points: jax.Array
    opponent_points: jax.Array
    unit_move_cost: jax.Array
    unit_sap_cost: jax.Array
    unit_sap_range: jax.Array
    unit_sensor_range: jax.Array
    agent_ids: jax.Array
 

def get_unit_embeddings(x, unit_positions):
    # x: (batch, height, width, emb_dim)
    # unit_positions: (batch, num_units, 2)
    batch_indices = jnp.arange(x.shape[0])[:, None]
    # Safeguarding against out-of-bounds indices
    return x[batch_indices, unit_positions[:, :, 0], unit_positions[:, :, 1]]

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


class ActorCritic(nn.Module):
    n_actions: int = 6
    info_emb_dim: int = 64
    hidden_dim: int = 256
 
    @nn.compact
    def __call__(self, actor_input):
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
            actor_input['observations'].reshape(-1, 10, 17, 17).transpose(0, 2, 3, 1)
        )

        state_embeddings = state_encoder(
            actor_input['states'].transpose((0, 2, 3, 1))
        )

        position_embeddings = get_2d_positional_embeddings(
            actor_input['positions'].reshape(-1, 2),
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
            state_embeddings.repeat(16, axis=0),
            info_embeddings.repeat(16, axis=0),
            env_info_embeddings.repeat(16, axis=0),
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


        critic_state_encoder = nn.Sequential(
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
        critic_state_embeddings = state_encoder(
            actor_input['states'].transpose((0, 2, 3, 1))
        )

        critic_info_embeddings = nn.Sequential([
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
        ])(info_input)

        embeddings = jnp.concat([
            critic_state_embeddings,
            critic_info_embeddings,
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

        return (logits1, logits2, logits3), values
