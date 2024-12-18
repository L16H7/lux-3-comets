# It throws error if I don't put it.
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

import jax
import jax.numpy as jnp
import orbax.checkpoint
import numpy as np

from representation import create_representations
from rnn import Actor, ScannedRNN

class DotDict:
    """A class that recursively converts dictionaries to objects 
    accessible with dot notation."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # Recursively convert nested dictionaries
                value = DotDict(value)
            setattr(self, key, value)

def reshape_observation(obs):
    obs = obs.replace(units=obs.units.replace(position=obs.units.position[None, :]))
    obs = obs.replace(units=obs.units.replace(energy=obs.units.energy[None, :]))
    obs = obs.replace(units_mask=obs.units_mask[None, :])
    obs = obs.replace(relic_nodes=obs.relic_nodes[None, :])
    obs = obs.replace(relic_nodes_mask=obs.relic_nodes_mask[None, :])
    obs = obs.replace(sensor_mask=obs.sensor_mask[None, :])
    obs = obs.replace(map_features=obs.map_features.replace(
        energy=obs.map_features.energy[None, :]))
    obs = obs.replace(map_features=obs.map_features.replace(
        tile_type=obs.map_features.tile_type[None, :]))
    obs = obs.replace(team_points=obs.team_points[None, :])
    obs = obs.replace(steps=jnp.atleast_1d(obs.steps))
    obs = obs.replace(match_steps=jnp.atleast_1d(obs.match_steps))
    return obs

def get_actions(logits):
    logits1, logits2, logits3 = logits
    action1 = np.argmax(logits1, axis=-1)
    action2 = np.argmax(logits2, axis=-1)
    action3 = np.argmax(logits3, axis=-1)

    return [action1, action2, action3]


class Agent():
    def __init__(self, player: str, env_cfg) -> None:
        self.player = player

        self.team_id = 0 if self.player == "player_0" else 1
        self.opponent_team_id = 1 if self.team_id == 0 else 0
        self.env_cfg = env_cfg

        checkpoint_path = os.path.join(script_dir, 'model_checkpoint')
        orbax_checkpointer = orbax.checkpoint.StandardCheckpointer()
        self.params = orbax_checkpointer.restore(checkpoint_path)

        self.actor = Actor(n_actions=6)
        self.inference_fn = jax.jit(lambda x, x1: self.actor.apply(x, x1))

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        obs = reshape_observation(DotDict(obs))

        relic_mask = obs.relic_nodes != -1
        self.discovered_relic_nodes[relic_mask] = obs.relic_nodes[relic_mask]

        representations = create_representations(
            tile_type=obs.map_features.tile_type,
            energy=obs.map_features.energy,
            sensor_mask=obs.sensor_mask,
            discovered_relic_nodes=self.discovered_relic_nodes,
            unit_masks_team=obs.units_mask[:, self.team_id, :],
            unit_masks_opponent=obs.units_mask[:, self.opp_team_id, :],
            unit_positions_team=obs.units.position[:, self.team_id, :, :],
            unit_energies_team=obs.units.energy[:, self.team_id, :],
            unit_positions_opponent=obs.units.position[:, self.opp_team_id, :, :],
            unit_energies_opponent=obs.units.energy[:, self.opp_team_id, :],
        )

        return jnp.zeros((16, 3), jnp.int32)
