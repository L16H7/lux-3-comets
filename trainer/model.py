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


class AttentionBlock(nn.Module):
    features: int
    num_heads: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    
    @nn.compact
    def __call__(self, x):
        B, H, W, C = x.shape
        residual = x
        
        # Reshape input to sequence: (B, H*W, C)
        x_seq = x.reshape(B, H * W, C)
        
        # Multi-head attention projections
        head_dim = self.features // self.num_heads
        scale = head_dim ** -0.5
        
        # Linear projections
        qkv = nn.Dense(features=3 * self.features)(x_seq)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, head_dim)
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        attn = (q @ jnp.transpose(k, (0, 1, 3, 2))) * scale
        attn = nn.softmax(attn, axis=-1)
        
        # Combine attention with values
        x = (attn @ v).transpose(0, 2, 1, 3)
        x = x.reshape(B, H * W, self.features)
        
        # Project back to original dimension
        x = nn.Dense(features=C)(x)
        
        # Reshape back to spatial dimensions
        x = x.reshape(B, H, W, C)
        
        # Add residual connection
        return x + residual


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
    point_gained_history: jax.Array
 
class Actor(nn.Module):
    n_actions: int = 6
    info_emb_dim: int = 256
    hidden_dim: int = 512
    position_emb_dim: int = 64
 
    @nn.compact
    def __call__(self, actor_input: ActorInput):
        observation_encoder = nn.Sequential([
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
            AttentionBlock(features=128, num_heads=8),
            nn.Conv(
                features=256,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(256),
            AttentionBlock(features=256, num_heads=8),
            nn.Conv(
                features=512,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=True
            ),
            nn.relu,
            ResidualBlock(512),
            nn.Conv(
                features=512,
                kernel_size=(8, 8),
                strides=(1, 1),
                padding=0,
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=True
            ),
            nn.relu,
            nn.Dense(512),
        ])


        observation_embeddings = observation_encoder(
            actor_input['observations'].transpose((0, 2, 3, 1))
        )

        position_embeddings = get_2d_positional_embeddings(
            actor_input['positions'],
            embedding_dim=self.position_emb_dim,
            max_size=24
        )

        info_input = jnp.concatenate([
            actor_input['team_points'][:, None],
            actor_input['opponent_points'][:, None],
            actor_input['match_steps'][:, None],
            actor_input['energies'].reshape(-1)[:, None],
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

        embeddings = jnp.concat([
            info_embeddings,
            position_embeddings,
            jnp.squeeze(observation_embeddings, axis=[1, 2]),
        ], axis=-1)

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
        seq_len, batch_size = critic_input['states'].shape[:2]

        state_encoder = nn.Sequential([
            nn.Conv(
                features=128,
                kernel_size=(4, 4),
                strides=(2, 2),
                padding='SAME',
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False,
            ),
            nn.relu,
            ResidualBlock(128),
            AttentionBlock(features=128, num_heads=8),
            nn.Conv(
                features=128,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding='SAME',
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(128),
            AttentionBlock(features=128, num_heads=8),
            nn.Conv(
                features=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(128),
            AttentionBlock(features=128, num_heads=8),
            nn.Conv(
                features=128,
                kernel_size=(3, 3),
                strides=(1, 1),
                padding='SAME',
                kernel_init=orthogonal(math.sqrt(2)),
                use_bias=False
            ),
            nn.relu,
            ResidualBlock(128),
            lambda x: x.reshape((x.shape[0], -1)),
            nn.Dense(512),
        ])

        state_embeddings = state_encoder(
            critic_input['states'].transpose((0, 2, 3, 1))
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
            nn.Dense(self.info_emb_dim, kernel_init=orthogonal(math.sqrt(2))),
            nn.relu,
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
                nn.relu,
                nn.Dense(1, kernel_init=orthogonal(1.0)),
            ]
        )

        values = critic(embeddings)
        return values
