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


class ActorInput(TypedDict):
    positions: jax.Array
    observations: jax.Array
    match_phases: jax.Array         # 4 phases each with 25 steps
    team_points: jax.Array
    opponent_points: jax.Array
    prev_points: jax.Array
    prev_actions: jax.Array
 
class Actor(nn.Module):
    n_actions: int = 6
    match_phase_emb_dim: int = 32
    match_emb_dim: int = 32
    point_info_emb_dim: int = 32
    action_emb_dim: int = 32
    hidden_dim: int = 256
    position_emb_dim: int = 32
 
    @nn.compact
    def __call__(self, hstate: jax.Array, actor_input: ActorInput):
        observation_encoder = nn.Sequential(
            [
                nn.Conv(
                    64,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    64,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    64,
                    (2, 2),
                    strides=1,
                    padding='SAME',
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    64,
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

        observation_embeddings = observation_encoder(actor_input['observations'])

        prev_action_embeddings = nn.Embed(
            self.n_actions,
            self.action_emb_dim
        )(actor_input['prev_actions'])

        match_phase_embeddings = nn.Embed(4, self.match_phase_emb_dim)(actor_input['match_phases'])

        point_info_embeddings = nn.Dense(self.point_info_emb_dim)(
            jnp.concat([
                actor_input['prev_points'],
                actor_input['team_points'],
                actor_input['opponent_points'],
            ], axis=-1)
        )

        position_embeddings = nn.Dense(self.position_emb_dim)(actor_input['positions'])

        embeddings = jnp.concat([
            position_embeddings,
            observation_embeddings,
            match_phase_embeddings,
            prev_action_embeddings,
            point_info_embeddings,
        ], axis=-1)

        actor = nn.Sequential(
            [
                nn.Dense(
                    self.hidden_dim, kernel_init=orthogonal(2),
                ),
                nn.leaky_relu,
            ]
        )

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
    hidden_dim: int = 512
 
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
                nn.Dense(512),
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
