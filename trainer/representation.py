import jax.numpy as jnp
import jax

from constants import Constants


NEBULA_TILE = 1
ASTEROID_TILE = 2

def transform_coordinates(coordinates, map_width=24, map_height=24):
    # Adjust for horizontal flip: (x, y) -> (MAP_WIDTH - 1 - x, y)
    flipped_positions = jnp.stack([map_width - 1 - coordinates[..., 0], coordinates[..., 1]], axis=-1)
    
    # Adjust for 90-degree rotation clockwise: (MAP_WIDTH - 1 - x, y) -> (y, MAP_WIDTH - 1 - x)
    rotated_positions = jnp.stack([map_height - 1 - flipped_positions[..., 1], flipped_positions[..., 0]], axis=-1)
    
    return rotated_positions

def transform_observation(obs):
    # Horizontal flip across the last dimension (24, 24 grids)
    flipped = jnp.flip(obs, axis=3)
    
    # Rotate 90 degrees clockwise after flip, across the last two dimensions (24x24)
    rotated = jnp.rot90(flipped, k=-1, axes=(2, 3))
    
    return rotated

def reconcile_positions(positions):
    first_half = positions[:, :3]
    last_half = positions[:, 3:]
    
    # Identify known and unknown positions (positions with -1 are considered unknown)
    is_known_first_half = jnp.all(first_half != -1, axis=-1)
    is_known_last_half = jnp.all(last_half != -1, axis=-1)
    
    # Transform coordinates for potential updates
    transformed_first_to_last = transform_coordinates(first_half)
    transformed_last_to_first = transform_coordinates(last_half)

    # Determine updates needed for unknown positions based on known positions
    update_first_from_last = jnp.where(is_known_last_half[:, :, None] & ~is_known_first_half[:, :, None],
                                       transformed_last_to_first, first_half)
    update_last_from_first = jnp.where(is_known_first_half[:, :, None] & ~is_known_last_half[:, :, None],
                                       transformed_first_to_last, last_half)

    # Concatenate the updated halves back together
    reconciled_positions = jnp.concatenate([update_first_from_last, update_last_from_first], axis=1)
    
    return reconciled_positions

def create_relic_nodes_maps(relic_nodes):
    n_envs, n_relic_nodes, _ = relic_nodes.shape
    relic_nodes_maps = jnp.zeros((n_envs, Constants.MAP_HEIGHT, Constants.MAP_WIDTH), dtype=jnp.int32)
    env_indices = jnp.repeat(jnp.arange(n_envs), n_relic_nodes)      # Shape: [n_envs * n_relic_nodes]

    relic_nodes_flat = relic_nodes.reshape(-1, 2)

    # Calculate possible positions directly
    relic_x_positions = relic_nodes_flat[:, 0].astype(jnp.int32)
    relic_y_positions = relic_nodes_flat[:, 1].astype(jnp.int32)

    relic_nodes_maps = relic_nodes_maps.at[env_indices, relic_y_positions, relic_x_positions].set(1)

    return relic_nodes_maps

def create_unit_maps(
    unit_positions,
    unit_masks,
    unit_energy,
):
    n_envs, n_units, _ = unit_positions.shape
    unit_maps = jnp.zeros((n_envs, Constants.MAP_HEIGHT, Constants.MAP_WIDTH), dtype=jnp.int32)
    unit_energy_maps = jnp.zeros((n_envs, Constants.MAP_HEIGHT, Constants.MAP_WIDTH), dtype=jnp.float32)
    env_indices = jnp.repeat(jnp.arange(n_envs), n_units)
    
    unit_positions_flat = unit_positions.reshape(-1, 2)
    unit_energy_flat = unit_energy.reshape(-1)
    unit_masks_flat = unit_masks.reshape(-1)
    
    unit_x_positions = unit_positions_flat[:, 0].astype(jnp.int32)
    unit_y_positions = unit_positions_flat[:, 1].astype(jnp.int32)
    
    unit_maps = unit_maps.at[env_indices, unit_y_positions, unit_x_positions].add(unit_masks_flat.astype(jnp.int32))
    unit_energy_maps = unit_energy_maps.at[env_indices, unit_y_positions, unit_x_positions].add(unit_energy_flat * unit_masks_flat.astype(jnp.int32))

    return unit_maps, unit_energy_maps

# @profile
def create_agent_patches(state_representation, unit_positions_team):
    n_envs, _, _, _ = state_representation.shape
    padding = ((0, 0),  # No padding for the batch dimension
               (0, 0),  # No padding for the channel dimension
               (4, 4),  # Pad height (top, bottom)
               (4, 4))  # Pad width (left, right)
    maps_padded = jnp.pad(state_representation, padding, mode='constant', constant_values=0)

    # Adjust agent positions to account for padding
    agent_positions_padded = unit_positions_team + 4  # Shape: (n_envs, n_agents_per_env, 2)

    # Generate environment indices for all agents
    n_agents_per_env = agent_positions_padded.shape[1]
    env_indices = jnp.repeat(jnp.arange(n_envs), n_agents_per_env)  # Repeat index for each agent

    # Flatten the positions to match the flat env_indices
    flat_positions = agent_positions_padded.reshape(-1, 2)  # Flatten positions

    @jax.jit
    def extract_patches_for_all_agents(maps_padded, flat_positions, env_indices):
        # Define a function to extract a patch for a single agent
        @jax.jit
        def extract_patch(env_idx, position):
            x, y = position.astype(jnp.int32)  # Ensure position indices are int32
            # Ensure all indices are of type int32
            start_indices = (env_idx.astype(jnp.int32), jnp.int32(0), y - 4, x - 4)
            slice_sizes = (1, state_representation.shape[1], 9, 9)
            patch = jax.lax.dynamic_slice(maps_padded, start_indices, slice_sizes)
            return patch.squeeze(0)  # Remove the batch dimension

        # Vectorize the extract_patch function over all agents
        patches = jax.vmap(extract_patch, in_axes=(0, 0))(env_indices, flat_positions)
        
        return patches

    # Call the function
    agent_patches = extract_patches_for_all_agents(maps_padded, flat_positions, env_indices)  # Shape: (n_envs * n_agents_per_env, 9, 9, 9)
    # Reshape back to (n_envs, n_agents_per_env, 9, 9, 9)
    agent_patches = agent_patches.reshape(n_envs, n_agents_per_env, state_representation.shape[1], 9, 9)

    return agent_patches


def create_representations(
    obs,
    discovered_relic_nodes,
    max_steps_in_match=100,
    team_idx=0,
    opponent_idx=1,
    points_gained=None,
):
    unit_masks_team = obs.units_mask[:, team_idx, :]              # Shape: [batch_size, num_team_units]
    unit_positions_team = obs.units.position[:, team_idx, :, :]   # Shape: [batch_size, num_team_units, 2]
    unit_energies_team = obs.units.energy[:, team_idx, :]         # Shape: [batch_size, num_team_units]

    unit_masks_opponent = obs.units_mask[:, opponent_idx, :]            # Shape: [batch_size, num_opponent_units]
    unit_positions_opponent = obs.units.position[:, opponent_idx, :, :] # Shape: [batch_size, num_opponent_units, 2]
    unit_energies_opponent = obs.units.energy[:, opponent_idx, :]       # Shape: [batch_size, num_opponent_units]

    relic_nodes = reconcile_positions(discovered_relic_nodes)

    team_unit_maps, team_energy_maps = create_unit_maps(
        unit_positions=unit_positions_team,
        unit_energy=unit_energies_team,
        unit_masks=unit_masks_team,
    )

    opponent_unit_maps, opponent_energy_maps = create_unit_maps(
        unit_positions=unit_positions_opponent,
        unit_energy=unit_energies_opponent,
        unit_masks=unit_masks_opponent,
    )

    relic_node_maps = create_relic_nodes_maps(relic_nodes=relic_nodes)

    asteroid_maps = jnp.where(obs.map_features.tile_type == ASTEROID_TILE, 1, 0)
    nebula_maps = jnp.where(obs.map_features.tile_type == NEBULA_TILE, 1, 0)

    # SCALE
    maps = [
        team_unit_maps / 8.0,
        team_energy_maps / 400.0,
        opponent_unit_maps / 8.0,
        opponent_energy_maps / 400.0,
        relic_node_maps,
        obs.map_features.energy.transpose((0, 2, 1)) / 20.0,
        asteroid_maps.transpose((0, 2, 1)),
        nebula_maps.transpose((0, 2, 1)),
        obs.sensor_mask.transpose((0, 2, 1)),
    ]

    state_representation = jnp.stack(maps, axis=1)
    state_representation = state_representation if team_idx == 0 else transform_observation(state_representation)

    match_phases = jnp.minimum(obs.match_steps[:, None] // 25, 3) # 4 phases
    matches = jnp.minimum(obs.steps[:, None] // max_steps_in_match, 4) # 5 matches
    team_points = obs.team_points if team_idx == 0 else jnp.flip(obs.team_points, axis=1)
    team_points = team_points / 400.0

    episode_info = jnp.concatenate((match_phases, matches, team_points), axis=1)

    unit_positions_team = unit_positions_team if team_idx == 0 else transform_coordinates(unit_positions_team)
    unit_positions_team = jnp.where(
        unit_positions_team == 24,
        -1,
        unit_positions_team,
    )
 
    unit_positions_opponent = unit_positions_opponent if team_idx == 0 else transform_coordinates(unit_positions_opponent)
    unit_positions_opponent = jnp.where(
        unit_positions_opponent == 24,
        -1,
        unit_positions_opponent,
    )
    
    agent_positions = (unit_positions_team + 1) / Constants.MAP_HEIGHT

    points_map = jnp.zeros((4, 24, 24))
    jax.debug.breakpoint()
    # agent_observations = create_agent_patches(
    #     state_representation=state_representation,
    #     unit_positions_team=unit_positions_team,
    # )
    # opponent_positions = (unit_positions_opponent + 1) / Constants.MAP_HEIGHT
    # relic_nodes_positions = (relic_nodes + 1) / Constants.MAP_HEIGHT

    return (
        state_representation,
        episode_info,
        agent_positions,
        unit_masks_team,
    )
        