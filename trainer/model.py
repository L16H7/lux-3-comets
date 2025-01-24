import jax
import math
import flax.linen as nn
import jax.numpy as jnp
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
 

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
    ) -> None:

        self.norm1 = nn.LayerNorm()
        self.attn = nn.MultiHeadDotProductAttention(
            kernel_init=nn.initializers.xavier_uniform(),
            broadcast_dropout=False,
            num_heads=num_heads
        )

        self.norm2 = nn.LayerNorm()

        self.mlp = nn.Sequential([
            nn.Dense(hidden_dim),
            nn.gelu,
            nn.Dense(hidden_dim),
        ])

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SEBlock(nn.Module):
    features: int
    reduction_ratio: int = 16

    @nn.compact
    def __call__(self, x):
        # Squeeze: Global Average Pooling
        squeezed = jnp.mean(x, axis=(1, 2), keepdims=True)
        
        # Excitation: Two fully connected layers with relu and sigmoid activations
        excitation = nn.Dense(features=self.features // self.reduction_ratio)(squeezed)
        excitation = nn.relu(excitation)
        excitation = nn.Dense(features=self.features)(excitation)
        excitation = nn.sigmoid(excitation)
        
        # Scale the input feature map
        return x * excitation


def get_unit_embeddings(x, unit_positions):
    # x: (batch, height, width, emb_dim)
    # unit_positions: (batch, num_units, 2)
    batch_indices = jnp.arange(x.shape[0])[:, None]
    # Safeguarding against out-of-bounds indices
    return x[batch_indices, unit_positions[:, :, 0], unit_positions[:, :, 1]]

class ResidualBlock(nn.Module):
    features: int
    kernel_size = (3, 3)
    strides = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = nn.Conv(self.features, self.kernel_size, self.strides, padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2)))(x)
        y = nn.leaky_relu(y, negative_slope=0.01)
        y = nn.Conv(self.features, self.kernel_size, self.strides, padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2)))(y)
        
        # Add Squeeze-and-Excitation layer
        y = SEBlock(self.features)(y)

        y += residual
        return nn.leaky_relu(y, negative_slope=0.01)


class ActorCritic(nn.Module):
    n_actions: int = 6
    info_emb_dim: int = 64
    hidden_dim: int = 256
 
    @nn.compact
    def __call__(self, actor_input):
        state_encoder = nn.Sequential([
            nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            ResidualBlock(32),
            nn.Conv(features=64, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            ResidualBlock(64),
            nn.Conv(features=128, kernel_size=(3, 3), strides=(2, 2), padding="SAME",
                    kernel_init=orthogonal(math.sqrt(2))),
            nn.leaky_relu,
            ResidualBlock(128),
        ])

        batch_size = actor_input['states'].shape[0]
        flat_info = jnp.concatenate([
            actor_input[k][:, None] for k in [
                'team_points', 'opponent_points', 'match_steps', 'matches', 
                'unit_move_cost', 'unit_sap_cost', 'unit_sap_range', 'unit_sensor_range'
            ]], axis=-1)
        info_input = flat_info.repeat(24 * 24, axis=1).reshape(batch_size, 24, 24, -1)

        states = jnp.concatenate([info_input, actor_input['states']], axis=-1)
        state_embeddings = state_encoder(states)

        action_head = nn.Conv(features=self.n_actions, kernel_size=(1, 1), padding="SAME",
                              kernel_init=orthogonal(math.sqrt(2)))
        x_coordinate_head = nn.Conv(features=17, kernel_size=(1, 1), padding="SAME",
                                    kernel_init=orthogonal(math.sqrt(2)))
        y_coordinate_head = nn.Conv(features=17, kernel_size=(1, 1), padding="SAME",
                                    kernel_init=orthogonal(math.sqrt(2)))

        logits_action = get_unit_embeddings(action_head(state_embeddings), actor_input['positions'])
        logits_x = get_unit_embeddings(x_coordinate_head(state_embeddings), actor_input['positions'])
        logits_y = get_unit_embeddings(y_coordinate_head(state_embeddings), actor_input['positions'])

        critic = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=orthogonal(2)),
            nn.relu,
            nn.Dense(1, kernel_init=orthogonal(1.0))
        ])
        values = critic(state_embeddings.mean(axis=(1, 2)))

        return (logits_action, logits_x, logits_y), values
