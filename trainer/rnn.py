import distrax
import functools
import jax
import math
import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import orthogonal
from typing import TypedDict


class ActorInput(TypedDict):
    observations: jax.Array
    prev_actions: jax.Array
    teams: jax.Array                # team 0 and team 1
    match_phases: jax.Array         # 4 phases each with 25 steps
    matches: jax.Array              # match number
    positions: jax.Array            # relative position of agent
    team_positions: jax.Array       # relative positions of all agents in a team
    opponent_positions: jax.Array   # relative positions of all opponent agents
    prev_rewards: jax.Array
    team_points: jax.Array
    opponent_points: jax.Array
    relic_nodes_positions: jax.Array
    

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


class Actor(nn.Module):
    n_actions: int = 5
    action_emb_dim: int = 8
    team_emb_dim: int = 8
    match_phase_emb_dim: int = 8
    match_emb_dim: int = 8
    position_emb_dim: int = 16
    relic_nodes_emb_dim: int = 32
    reward_info_emb_dim: int = 16
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, hstate: jax.Array, actor_input: ActorInput):
        observation_encoder = nn.Sequential(
            [
                nn.Conv(
                    64,
                    (2, 2),
                    strides=1,
                    padding=0,
                    kernel_init=orthogonal(math.sqrt(2)),
                ),
                nn.leaky_relu,
                nn.Conv(
                    64,
                    (2, 2),
                    strides=2,
                    padding=0,
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

        team_embeddings = nn.Embed(2, self.team_emb_dim)(actor_input['teams'])
        match_phase_embeddings = nn.Embed(4, self.match_phase_emb_dim)(actor_input['match_phases'])
        match_embeddings = nn.Embed(5, self.match_emb_dim)(actor_input['matches'])

        position_embeddings = nn.Dense(self.position_emb_dim)(actor_input['positions'])
        team_position_embeddings = nn.Dense(self.position_emb_dim)(
            actor_input['team_positions'].reshape(
                actor_input['team_positions'].shape[0],
                actor_input['team_positions'].shape[1],
                -1,
            )
        )
        opponent_position_embeddings = nn.Dense(self.position_emb_dim)(
            actor_input['opponent_positions'].reshape(
                actor_input['opponent_positions'].shape[0],
                actor_input['opponent_positions'].shape[1],
                -1,
            )
        )

        relic_nodes_embeddings = nn.Dense(self.relic_nodes_emb_dim)(
            actor_input['relic_nodes_positions'].reshape(
                actor_input['relic_nodes_positions'].shape[0],
                actor_input['relic_nodes_positions'].shape[1],
                -1,
            )
        )

        reward_info_embeddings = nn.Dense(self.reward_info_emb_dim)(
            jnp.concat([
                actor_input['prev_rewards'],
                actor_input['team_points'],
                actor_input['opponent_points'],
            ], axis=-1)
        )

        embeddings = jnp.concat([
            observation_embeddings,
            prev_action_embeddings,
            team_embeddings,
            match_phase_embeddings,
            match_embeddings,
            position_embeddings,
            relic_nodes_embeddings,
            team_position_embeddings,
            opponent_position_embeddings,
            reward_info_embeddings,
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
        logits2 = nn.Dense(9, kernel_init=orthogonal(0.01))(x)
        logits3 = nn.Dense(9, kernel_init=orthogonal(0.01))(x)
       
        return [logits1, logits2, logits3], hstate


class CriticInput(TypedDict):
    states: jax.Array
    teams: jax.Array                # team 0 and team 1
    match_phases: jax.Array         # 4 phases each with 25 steps
    matches: jax.Array              # match number
    team_points: jax.Array
    opponent_points: jax.Array
 

class Critic(nn.Module):
    team_emb_dim: int = 16
    match_phase_emb_dim: int = 16
    match_emb_dim: int = 16
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

        team_embeddings = nn.Embed(2, self.team_emb_dim)(critic_input['teams'])
        match_phase_embeddings = nn.Embed(4, self.match_phase_emb_dim)(critic_input['match_phases'])
        match_embeddings = nn.Embed(5, self.match_emb_dim)(critic_input['matches'])

        point_info_embeddings = nn.Dense(self.point_info_emb_dim)(
            jnp.concat([
                critic_input['team_points'],
                critic_input['opponent_points'],
            ], axis=-1)
        )

        embeddings = jnp.concat([
            state_embeddings,
            team_embeddings,
            match_phase_embeddings,
            match_embeddings,
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
