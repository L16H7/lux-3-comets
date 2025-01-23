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


def get_unit_embeddings(x, unit_positions):
    # x: (batch, 24, 24, emb_dim)
    # unit_positions: (batch, num_units, 2)
    batch_indices = jnp.arange(x.shape[0])[:, None]
    return x[batch_indices, unit_positions[:, :, 0], unit_positions[:, :, 1]]


class ActorCritic(nn.Module):
    n_actions: int = 6
    info_emb_dim: int = 64
    hidden_dim: int = 256
 
    @nn.compact
    def __call__(self, actor_input: ActorInput):
        state_patch_encoder = nn.Conv(
            features=32,
            kernel_size=(4, 4),
            strides=(4, 4),
            kernel_init=orthogonal(math.sqrt(2))
        )

        batch_size = actor_input['states'].shape[0]

        patch_embeddings = state_patch_encoder(
            actor_input['states']
        )

        patch_embeddings = patch_embeddings.reshape(batch_size, -1, 32)

        seq_len = patch_embeddings.shape[1]
        positional_embeddings = sinusoidal_positional_encoding(seq_len, 32)
        embeddings = patch_embeddings + positional_embeddings[None, :, :]
        
        info_input = jnp.stack([
            actor_input['team_points'],
            actor_input['opponent_points'],
            actor_input['match_steps'],
            actor_input['matches'],
            actor_input['unit_move_cost'],
            actor_input['unit_sap_cost'],
            actor_input['unit_sap_range'],
            actor_input['unit_sensor_range'],
        ], axis=-1)

        info_embeddings = nn.Sequential([
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.relu,
        ])(info_input)

        x = jnp.concatenate([
            embeddings,
            info_embeddings[:, None, :].repeat(seq_len, axis=1)
        ], axis=-1)

        encoder = TransformerEncoder(
            hidden_dim=96,
            num_heads=4
        )
        state_embeddings = encoder(x)
        upsample_layer = nn.ConvTranspose(
            features=self.hidden_dim, 
            kernel_size=(4, 4), 
            strides=(4, 4),
            kernel_init=orthogonal(math.sqrt(2))
        )
        x = upsample_layer(state_embeddings.reshape(batch_size, 6, 6, -1))
        x = x.reshape(batch_size, 24, 24, -1)

        unit_embeddings = get_unit_embeddings(x, actor_input['positions'])

        actor = nn.Sequential(
            [
                nn.Dense(
                    256, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dense(
                    256, kernel_init=orthogonal(2),
                ),
                nn.relu,
            ]
        )

        x = actor(unit_embeddings)

        action_head = nn.Dense(self.n_actions, kernel_init=orthogonal(0.01))

        x_coordinate_head = nn.Dense(17, kernel_init=orthogonal(0.01))
        y_coordinate_head = nn.Dense(17, kernel_init=orthogonal(0.01))

        logits1 = action_head(x)
        logits2 = x_coordinate_head(x)
        logits3 = y_coordinate_head(x)

        critic = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        values = critic(state_embeddings.reshape(batch_size, -1))

        return (logits1, logits2, logits3), values
