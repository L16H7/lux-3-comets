import jax.numpy as jnp
import jax


NEBULA_TILE = 1
ASTEROID_TILE = 2

def create_relic_nodes_maps(relic_nodes, relic_nodes_mask, map_height, map_width):
    n_envs, n_relic_nodes, _ = relic_nodes.shape
    relic_nodes_maps = jnp.zeros((n_envs, map_height, map_width), dtype=jnp.int32)
    env_indices = jnp.repeat(jnp.arange(n_envs), n_relic_nodes)      # Shape: [n_envs * n_relic_nodes]

    # Flatten relic nodes and their masks
    relic_nodes_flat = relic_nodes.reshape(-1, 2)
    relic_nodes_mask_flat = relic_nodes_mask.reshape(-1)

    # Calculate possible positions directly
    relic_x_positions = relic_nodes_flat[:, 0].astype(jnp.int32)
    relic_y_positions = relic_nodes_flat[:, 1].astype(jnp.int32)

    # Apply mask directly in the assignment to avoid dynamic shapes
    relic_nodes_maps = relic_nodes_maps.at[env_indices, relic_y_positions, relic_x_positions].add(relic_nodes_mask_flat.astype(jnp.int32))

    return relic_nodes_maps

def create_unit_maps(
    unit_positions,
    unit_masks,
    unit_energy,
    map_width,
    map_height,
):
    n_envs, n_units, _ = unit_positions.shape
    unit_maps = jnp.zeros((n_envs, map_height, map_width), dtype=jnp.int32)
    unit_energy_maps = jnp.zeros((n_envs, map_height, map_width), dtype=jnp.float32)
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
    map_height,
    map_width,
    max_steps_in_match=100,
    team_idx=0,
    enemy_idx=1,
):
    unit_masks_team = obs.units_mask[:, team_idx, :]              # Shape: [batch_size, num_team_units]
    unit_positions_team = obs.units.position[:, team_idx, :, :]   # Shape: [batch_size, num_team_units, 2]
    unit_energies_team = obs.units.energy[:, team_idx, :]         # Shape: [batch_size, num_team_units]

    unit_masks_enemy = obs.units_mask[:, enemy_idx, :]            # Shape: [batch_size, num_enemy_units]
    unit_positions_enemy = obs.units.position[:, enemy_idx, :, :] # Shape: [batch_size, num_enemy_units, 2]
    unit_energies_enemy = obs.units.energy[:, enemy_idx, :]       # Shape: [batch_size, num_enemy_units]

    relic_nodes = discovered_relic_nodes
    # relic_nodes_mask = obs.relic_nodes_mask
    relic_nodes_mask = (discovered_relic_nodes[:, :, 0] != -1)

    team_unit_maps, team_energy_maps = create_unit_maps(
        unit_positions=unit_positions_team,
        unit_energy=unit_energies_team,
        unit_masks=unit_masks_team,
        map_width=map_width,
        map_height=map_height,
    )

    enemy_unit_maps, enemy_energy_maps = create_unit_maps(
        unit_positions=unit_positions_enemy,
        unit_energy=unit_energies_enemy,
        unit_masks=unit_masks_enemy,
        map_width=map_width,
        map_height=map_height,
    )


    relic_node_maps = create_relic_nodes_maps(
        relic_nodes=relic_nodes,
        relic_nodes_mask=relic_nodes_mask,
        map_width=map_width,
        map_height=map_height,
    )

    asteroid_maps = jnp.where(obs.map_features.tile_type == ASTEROID_TILE, 1, 0)
    nebula_maps = jnp.where(obs.map_features.tile_type == NEBULA_TILE, 1, 0)

    # SCALE
    maps = [
        team_unit_maps / 16.0,
        team_energy_maps / 800.0,
        enemy_unit_maps / 16.0,
        enemy_energy_maps / 800.0,
        relic_node_maps,
        obs.map_features.energy.transpose((0, 2, 1)) / 20.0,
        asteroid_maps.transpose((0, 2, 1)),
        nebula_maps.transpose((0, 2, 1)),
        obs.sensor_mask.transpose((0, 2, 1)),
    ]

    state_representation = jnp.stack(maps, axis=1)

    match_phases = jnp.minimum(obs.match_steps[:, None] // 25, 3) # 4 phases
    matches = jnp.minimum(obs.steps[:, None] // max_steps_in_match, 4) # 5 matches
    teams = jnp.ones_like(match_phases, dtype=jnp.int32) * team_idx
    team_points = obs.team_points if team_idx == 0 else jnp.flip(obs.team_points, axis=1)
    team_points = team_points / 200.0

    episode_info = jnp.concatenate((teams, match_phases, matches, team_points), axis=1)
    
    agent_observations = create_agent_patches(
        state_representation=state_representation,
        unit_positions_team=unit_positions_team,
    )
    agent_positions = (unit_positions_team + 1) / map_height
    opponent_positions = (unit_positions_enemy + 1) / map_height
    relic_nodes_positions = (discovered_relic_nodes + 1) / map_height

    return (
        state_representation,
        agent_observations,
        episode_info,
        agent_positions,
        opponent_positions,
        relic_nodes_positions,
        unit_masks_team,
    )
        