import jax
import math
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import orthogonal
from typing import Tuple, TypedDict


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
    positions = (positions / (max_size / 2)) - 1
    
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
    energies: jax.Array
    energies_gained: jax.Array
    point_gained_history: jax.Array
 
class Actor(nn.Module):
    n_actions: int = 6
    hidden_dim: int = 786
    position_emb_dim: int = 64
    info_emb_dim = 386
    n_heads: int = 8
 
    @nn.compact
    def __call__(self, actor_input: ActorInput):
        observation_encoder = nn.Sequential([
            nn.Conv(
                features=64,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False,
            ),
            nn.relu,
            ResidualBlock(64),
            nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(64),
            nn.Conv(
                features=64,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(64),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(512),
            nn.relu,
        ])


        observation_embeddings = observation_encoder(
            actor_input['observations'].transpose((0, 2, 3, 1))
        )

        position_embeddings = get_2d_positional_embeddings(
            actor_input['positions'],
            embedding_dim=32,
            max_size=24
        )

        info_input = jnp.concatenate([
            actor_input['team_points'][:, None],
            actor_input['opponent_points'][:, None],
            actor_input['match_steps'][:, None],
            actor_input['energies'].reshape(-1)[:, None],
            actor_input['energies_gained'].reshape(-1)[:, None],
            actor_input['unit_sap_cost'][:, None],
            actor_input['unit_sap_range'][:, None],
            actor_input['points_gained_history']
        ], axis=-1)

        info_embeddings = nn.Sequential([
            nn.Dense(
                self.hidden_dim, kernel_init=orthogonal(2),
            ),
            nn.relu,
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.relu,
        ])(info_input)

        actor = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
            ]
        )

        embeddings = jnp.concat([
            info_embeddings,
            position_embeddings,
            observation_embeddings,
        ], axis=-1)

        x = actor(embeddings)

        action_head = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dense(
                    self.n_actions, kernel_init=orthogonal(0.01),
                ),
            ]
        )

        x_coordinate_head = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dense(
                    17, kernel_init=orthogonal(0.01),
                ),
            ]
        )

        y_coordinate_head = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dense(
                    17, kernel_init=orthogonal(0.01),
                ),
            ]
        )

        logits1 = action_head(x)
        logits2 = x_coordinate_head(x)
        logits3 = y_coordinate_head(x)

        return logits1, logits2, logits3


class ResidualBlock(nn.Module):
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    reduction_ratio: int = 16  # Reduction ratio for SE block
    
    @nn.compact
    def __call__(self, x):
        residual = x
        
        # Main convolution path
        y = nn.Conv(self.features, self.kernel_size, self.strides, 
                   padding="SAME", use_bias=False)(x)
        y = nn.relu(y)
        y = nn.Conv(self.features, self.kernel_size, self.strides, 
                   padding="SAME", use_bias=False)(y)
        
        # Squeeze and Excitation block
        # Squeeze: Global average pooling
        se = jnp.mean(y, axis=(1, 2), keepdims=True)  # Shape: (B, 1, 1, C)
        
        # Excitation: Two FC layers with reduction
        se_features = max(self.features // self.reduction_ratio, 1)
        se = nn.Conv(se_features, (1, 1), use_bias=True)(se)  # First FC
        se = nn.relu(se)
        se = nn.Conv(self.features, (1, 1), use_bias=True)(se)  # Second FC
        se = nn.sigmoid(se)
        
        # Scale the original features
        y = y * se
        
        # Add residual connection
        y += residual
        return nn.relu(y)
    

class CriticInput(TypedDict):
    states: jax.Array
    match_steps: jax.Array         # 4 phases each with 25 steps
    team_points: jax.Array
    opponent_points: jax.Array
 

class Critic(nn.Module):
    info_emb_dim: int = 96
    hidden_dim: int = 512
 
    @nn.compact
    def __call__(self, critic_input):
        state_encoder = nn.Sequential([
            nn.Conv(
                features=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False,
            ),
            nn.relu,
            ResidualBlock(128),
            nn.Conv(
                features=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(128),
            nn.Conv(
                features=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(128),
            nn.Conv(
                features=512,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
        ])

        state_embeddings = state_encoder(
            critic_input['states']
        )

        info_input = jnp.concatenate([
            critic_input['team_points'][:, None],
            critic_input['opponent_points'][:, None],
            critic_input['match_steps'][:, None],
            critic_input['matches'][:, None],
            critic_input['points_gained_history'],
        ], axis=-1)


        info_embeddings = nn.Sequential([
            nn.Dense(self.hidden_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.relu,
            nn.Dropout(rate=0.2, deterministic=False),
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.relu,
        ])(info_input)

        embeddings = jnp.concat([
            jnp.squeeze(state_embeddings, axis=[1, 2]),
            info_embeddings,
        ], axis=-1)

        critic = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.relu,
                nn.Dropout(rate=0.2, deterministic=False),
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        values = critic(embeddings)
        return values
