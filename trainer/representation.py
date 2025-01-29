import jax.numpy as jnp
import jax

from constants import Constants
from points import update_points_map_batch, mark_duplicates_batched, filter_by_proximity_batch


NEBULA_TILE = 1
ASTEROID_TILE = 2

def transform_coordinates(coordinates, map_width=24, map_height=24):
    # Adjust for horizontal flip: (x, y) -> (MAP_WIDTH - 1 - x, y)
    flipped_positions = jnp.stack([map_width - 1 - coordinates[..., 0], coordinates[..., 1]], axis=-1)
    
    # Adjust for 90-degree rotation clockwise: (MAP_WIDTH - 1 - x, y) -> (y, MAP_WIDTH - 1 - x)
    rotated_positions = jnp.stack([map_height - 1 - flipped_positions[..., 1], flipped_positions[..., 0]], axis=-1)
    
    return rotated_positions

def transform_observation_3dim(obs):
    # Horizontal flip across the last dimension (24, 24 grids)
    flipped = jnp.flip(obs, axis=2)
    
    # Rotate 90 degrees clockwise after flip, across the last two dimensions (24x24)
    rotated = jnp.rot90(flipped, k=-1, axes=(1, 2))
    
    return rotated

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

def create_relic_nodes_maps(relic_nodes, relic_nodes_mask):
    n_envs, n_relic_nodes, _ = relic_nodes.shape
    relic_nodes_maps = jnp.zeros((n_envs, Constants.MAP_HEIGHT, Constants.MAP_WIDTH), dtype=jnp.int32)
    env_indices = jnp.repeat(jnp.arange(n_envs), n_relic_nodes)      # Shape: [n_envs * n_relic_nodes]

    relic_nodes_flat = relic_nodes.reshape(-1, 2)
    relic_nodes_mask_flat = relic_nodes_mask.reshape(-1)

    # Calculate possible positions directly
    relic_x_positions = relic_nodes_flat[:, 0].astype(jnp.int32)
    relic_y_positions = relic_nodes_flat[:, 1].astype(jnp.int32)

    relic_nodes_maps = relic_nodes_maps.at[env_indices, relic_y_positions, relic_x_positions].add(relic_nodes_mask_flat.astype(jnp.int32))

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
    side = 23
    full = (side * 2) + 1
    n_envs, _, _, _ = state_representation.shape
    padding = ((0, 0),  # No padding for the batch dimension
               (0, 0),  # No padding for the channel dimension
               (side, side),  # Pad height (top, bottom)
               (side, side))  # Pad width (left, right)
    maps_padded = jnp.pad(state_representation, padding, mode='constant', constant_values=0)

    # Adjust agent positions to account for padding
    agent_positions_padded = unit_positions_team + side  # Shape: (n_envs, n_agents_per_env, 2)

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
            start_indices = (env_idx.astype(jnp.int32), jnp.int32(0), y - side, x - side)
            slice_sizes = (1, state_representation.shape[1], full, full)
            patch = jax.lax.dynamic_slice(maps_padded, start_indices, slice_sizes)
            return patch.squeeze(0)  # Remove the batch dimension

        # Vectorize the extract_patch function over all agents
        patches = jax.vmap(extract_patch, in_axes=(0, 0))(env_indices, flat_positions)
        
        return patches

    # Call the function
    agent_patches = extract_patches_for_all_agents(maps_padded, flat_positions, env_indices)  # Shape: (n_envs * n_agents_per_env, 9, 9, 9)
    # Reshape back to (n_envs, n_agents_per_env, 9, 9, 9)
    agent_patches = agent_patches.reshape(n_envs, n_agents_per_env, state_representation.shape[1], full, full)

    return agent_patches


def get_env_info(env_params):
    unit_move_cost = jnp.expand_dims(
        env_params.unit_move_cost, axis=-1
    ).repeat(16, axis=1) / 8.0
    unit_sap_cost = (jnp.expand_dims(
        env_params.unit_sap_cost, axis=-1
    ).repeat(16, axis=1) - 30.0) / 20.0
    unit_sap_range = jnp.expand_dims(
        env_params.unit_sap_range, axis=-1
    ).repeat(16, axis=1) / 8.0
    unit_sensor_range = jnp.expand_dims(
        env_params.unit_sensor_range, axis=-1
    ).repeat(16, axis=1) / 8.0

    env_info = jnp.concatenate([
        unit_move_cost[..., None],
        unit_sap_cost[..., None],
        unit_sap_range[..., None],
        unit_sensor_range[..., None]
    ], axis=-1)
    return env_info.reshape(-1, 4)


def create_representations(
    obs,
    temporal_states,
    discovered_relic_nodes,
    prev_agent_positions,
    points_map,
    search_map,
    points_gained,
    max_steps_in_match=100,
    team_idx=0,
    opponent_idx=1,
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

    relic_node_maps = create_relic_nodes_maps(
        relic_nodes=relic_nodes,
        relic_nodes_mask=relic_nodes[..., 0] != -1,
    )

    asteroid_maps = jnp.where(obs.map_features.tile_type == ASTEROID_TILE, 1, 0)
    nebula_maps = jnp.where(obs.map_features.tile_type == NEBULA_TILE, 1, 0)

    # Update points map
    proximity_positions = filter_by_proximity_batch(
        prev_agent_positions,
        relic_nodes
    )
    # prev_agent_positions = jnp.where(
    #     points_gained[:, None, None] > 0,
    #     proximity_positions,
    #     prev_agent_positions,
    # )
    prev_agent_positions = proximity_positions


    points_map = points_map.at[:, 0, 0].set(-1)
    points_map = points_map.at[:, -1, -1].set(-1)
    updated_points_map = update_points_map_batch(
        points_map,
        mark_duplicates_batched(prev_agent_positions),
        points_gained,
    )
    transformed_updated_points_map = transform_observation_3dim(updated_points_map)

    updated_points_map = jnp.where(
        updated_points_map != 0,
        updated_points_map,
        transformed_updated_points_map,
    )

    updated_points_map = jnp.where(
        obs.steps[0] == 102,
        jnp.maximum(updated_points_map, 0),
        updated_points_map
    )
    updated_points_map = jnp.where(
        obs.steps[0] == 203,
        jnp.maximum(updated_points_map, 0),
        updated_points_map
    )

    updated_points_map = jnp.where(
        obs.steps[0] == 506,
        jnp.zeros_like(updated_points_map),
        updated_points_map
    )

    sensor_mask = obs.sensor_mask.transpose((0, 2, 1))
    updated_search_map = jnp.where(
        sensor_mask,
        1,
        search_map
    )
 
    transformed_sensor_mask = transform_observation_3dim(sensor_mask)
    updated_search_map = jnp.where(
        transformed_sensor_mask,
        1,
        updated_search_map
    )

    updated_search_map = jnp.where(
        obs.steps[0] == 102,
        0,
        updated_search_map
    )
    updated_search_map = jnp.where(
        obs.steps[0] == 203,
        0,
        updated_search_map
    )
    updated_search_map = jnp.where(
        obs.steps[0] == 506,
        0,
        updated_search_map
    )

    # if relic_nodes are found for match 1
    updated_search_map = jnp.where(
        jnp.logical_and(
            (relic_nodes[..., 0] > -1).sum(axis=-1) == 2,
            obs.steps < 101
        )[:, None, None],
        jnp.ones_like(updated_search_map),
        updated_search_map
    )

    # if relic_nodes are found for match 2
    updated_search_map = jnp.where(
        jnp.logical_and(
            (relic_nodes[..., 0] > -1).sum(axis=-1) == 4,
            obs.steps < 202
        )[:, None, None],
        jnp.ones_like(updated_search_map),
        updated_search_map
    )

    # if relic_nodes are found for match 3
    updated_search_map = jnp.where(
        jnp.logical_and(
            (relic_nodes[..., 0] > -1).sum(axis=-1) == 6,
            obs.steps < 303
        )[:, None, None],
        jnp.ones_like(updated_search_map),
        updated_search_map
    )

    # if relic_nodes doesn't spawn in match 2 (or if we suck and don't find)
    updated_search_map = jnp.where(
        jnp.logical_and(
            (discovered_relic_nodes[..., 0] > -1).sum(axis=-1) == 2,
            obs.steps > 202
        )[:, None, None],
        jnp.ones_like(updated_search_map),
        updated_search_map
    )

    energy_map = jnp.where(
        obs.sensor_mask,
        obs.map_features.energy,
        0
    )
    maps = [
        team_energy_maps / 800.0,
        opponent_energy_maps / 800.0,
        asteroid_maps.transpose((0, 2, 1)),
        energy_map.transpose((0, 2, 1)) / 20.0,
        nebula_maps.transpose((0, 2, 1)),
        obs.sensor_mask.transpose((0, 2, 1)),
        relic_node_maps,
        updated_points_map,
        updated_search_map,
        team_unit_maps / 4.0,
        opponent_unit_maps / 4.0,
    ]
    state_representation = jnp.stack(maps, axis=1)
    state_representation = state_representation if team_idx == 0 else transform_observation(state_representation)


    match_steps = obs.match_steps[:, None] / 100.0
    matches = jnp.minimum(obs.steps[:, None] // max_steps_in_match, 4) / 4.0
    team_points = obs.team_points if team_idx == 0 else jnp.flip(obs.team_points, axis=1)
    team_points = team_points / 800.0

    episode_info = jnp.concatenate((match_steps, matches, team_points), axis=1)

    transformed_unit_positions = transform_coordinates(unit_positions_team)
    transformed_unit_positions = jnp.where(
        transformed_unit_positions == 24,
        -1,
        transformed_unit_positions,
    )
    unit_positions_team = unit_positions_team if team_idx == 0 else transformed_unit_positions
 
    agent_positions = unit_positions_team

    agent_observations = create_agent_patches(
        state_representation=jnp.concatenate([temporal_states, state_representation], axis=1),
        unit_positions_team=unit_positions_team,
    )

    updated_temporal_states = jnp.concatenate([
        temporal_states[:, 3:, ...],
        state_representation[:, :3, ...],
    ], axis=1)

    return (
        state_representation,
        updated_temporal_states,
        agent_observations,
        episode_info,
        updated_points_map,
        updated_search_map,
        agent_positions,
        unit_energies_team / 400.0,
        unit_energies_team > 0, # mask energy depleted agents in ppo update
        discovered_relic_nodes,
    )
        
def create_agent_representations(
    observations,
    p0_temporal_states,
    p1_temporal_states,
    p0_discovered_relic_nodes,
    p1_discovered_relic_nodes,
    p0_points_map,
    p1_points_map,
    p0_search_map,
    p1_search_map,
    p0_points_gained,
    p1_points_gained,
    p0_prev_agent_positions,
    p1_prev_agent_positions,
):
    p0_observations = observations["player_0"]
    p0_representations = create_representations(
        obs=p0_observations,
        temporal_states=p0_temporal_states,
        discovered_relic_nodes=p0_discovered_relic_nodes,
        prev_agent_positions=p0_prev_agent_positions,
        points_map=p0_points_map,
        search_map=p0_search_map,
        points_gained=p0_points_gained,
        team_idx=0,
        opponent_idx=1,
    )

    p1_observations = observations["player_1"]
    p1_representations = create_representations(
        obs=p1_observations,
        temporal_states=p1_temporal_states,
        discovered_relic_nodes=p1_discovered_relic_nodes,
        prev_agent_positions=p1_prev_agent_positions,
        points_map=p1_points_map,
        search_map=p1_search_map,
        points_gained=p1_points_gained,
        team_idx=1,
        opponent_idx=0,
    )
    return p0_representations, p1_representations

def combined_states_info(team_states, opponent_states):
    opponent_states = transform_observation(opponent_states.copy())
    combined_states = jnp.stack([
        team_states[:, 0, ...], 
        opponent_states[:, 0, ...],
        # energy
        jnp.where(team_states[:, 3, ...] != 0, team_states[:, 3, ...], opponent_states[:, 3, ...]),
        # asteroid
        jnp.where(team_states[:, 2, ...] != 0, team_states[:, 2, ...], opponent_states[:, 2, ...]),
        # nebula
        jnp.where(team_states[:, 4, ...] != 0, team_states[:, 4, ...], opponent_states[:, 4, ...]),
        team_states[:, 6, ...], # relic
        opponent_states[:, 6, ...], # relic
        team_states[:, 7, ...], # point map
        opponent_states[:, 7, ...], # point map
        team_states[:, 8, ...], # search_map
        team_states[:, 9, ...], # team units
        opponent_states[:, 9, ...] # opponent units
    ], axis=1)

    return combined_states
