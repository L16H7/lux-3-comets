import jax
import distrax

import jax.numpy as jnp

from constants import Constants
from representation import transform_coordinates, reconcile_positions


directions = jnp.array(
    [
        [0, -1],    # Move up
        [1, 0],     # Move right
        [0, 1],     # Move down
        [-1, 0],    # Move left
    ],
    dtype=jnp.int16,
)

def transform_observation(obs):
    # Horizontal flip across the last dimension (24, 24 grids)
    flipped = jnp.flip(obs, axis=2)
    
    # Rotate 90 degrees clockwise after flip, across the last two dimensions (24x24)
    rotated = jnp.rot90(flipped, k=-1, axes=(1, 2))
    
    return rotated

def vectorized_transform_actions(actions):
    # Create a JAX array that maps each action index to its new action
    # Index:      0  1  2  3  4 . 5
    # Action map: 0  2  1  4  3 . 5
    action_map = jnp.array([0, 2, 1, 4, 3, 5])

    # Vectorized mapping
    transformed_actions = action_map[actions]
    return transformed_actions

def mask_sap_range(logits_slice, cutoff_range):
    cols = logits_slice.shape[1]
    mask = jnp.arange(cols) < cutoff_range
    logits_slice = jnp.where(mask[None, :], 0, logits_slice)
    
    mask2 = jnp.arange(cols) > (16 - cutoff_range)
    logits_slice = jnp.where(mask2[None, :], 0, logits_slice)
    return logits_slice

mask_sap_range_vmap = jax.vmap(
    mask_sap_range,
    in_axes=(0, 0)
)

def mask_out_of_bounds(agent_positions):
    target_coods = jnp.arange(-8, 9)
    target_x = agent_positions.reshape(-1, 2)[:, 0][:, None] + target_coods[None, :]
    target_x = (target_x >= 0) & (target_x < Constants.MAP_WIDTH)

    target_y = agent_positions.reshape(-1, 2)[:, 1][:, None] + target_coods[None, :]
    target_y = (target_y >= 0) & (target_y < Constants.MAP_HEIGHT)
    return target_x, target_y

def filter_targets_with_boolean_map(targets, boolean_map):
    """
    Filter target positions, replacing with (-1, -1) if boolean_map is True at that position.
    
    Args:
        target_positions (jnp.ndarray): Shape (n_envs, 16, 2) array of target positions
        boolean_mask (jnp.ndarray): Shape (n_envs, 24, 24) boolean array
        
    Returns:
        jnp.ndarray: Shape (n_envs, 16, 2) filtered target positions
    """
    x_indices, y_indices = targets[..., 0], targets[..., 1]
    
    # Get the boolean values from maps using advanced indexing
    result = boolean_map[jnp.arange(boolean_map.shape[0])[:, None, None], y_indices, x_indices]
    return result


def compute_collision_avoidance(ally_pos, ally_energy, enemy_pos, enemy_energy, directions):
    """
    Computes an action mask for allied agents given positions and energies.
    
    Parameters:
      ally_pos:    jnp.array with shape (n_envs, 16, 2)
      ally_energy: jnp.array with shape (n_envs, 16)
      enemy_pos:   jnp.array with shape (n_envs, 16, 2)
      enemy_energy:jnp.array with shape (n_envs, 16)
      directions:  jnp.array with shape (5, 2)  -- first row is center, others are moves
      
    Returns:
      final_mask:  jnp.array with shape (n_envs, 16, 5)
                   A boolean mask where True indicates an allowed action.
    """
    # 1. Compute candidate positions for each allied agent and each action.
    #    Shape: (n_envs, 16, 5, 2)
    candidate_positions = ally_pos[:, :, None, :] + directions[None, None, :, :]

    # 2. Compare candidate positions with enemy positions.
    #    We want to know for each candidate position (for each ally and action) whether
    #    an enemy is exactly at that location.
    #    Expand enemy positions to shape: (n_envs, 1, 1, 16, 2)
    enemy_pos_exp = enemy_pos[:, None, None, :, :]
    
    #    Check equality along the last dimension (both coordinates must match).
    #    The result 'is_same' has shape: (n_envs, 16, 5, 16)
    is_same = jnp.all(candidate_positions[:, :, :, None, :] == enemy_pos_exp, axis=-1)

    # 3. Check the energy condition.
    #    Expand energies for proper broadcasting:
    #      - ally_energy: (n_envs, 16) -> (n_envs, 16, 1, 1)
    #      - enemy_energy: (n_envs, 16) -> (n_envs, 1, 1, 16)
    ally_energy_exp = ally_energy[:, :, None, None]
    enemy_energy_exp = enemy_energy[:, None, None, :]

    #    Determine where an enemy has higher energy than the ally.
    #    Shape: (n_envs, 16, 1, 16)
    enemy_stronger = enemy_energy_exp > ally_energy_exp

    #    Combine spatial match and energy condition.
    #    For each candidate move and enemy, threat_per_enemy is True if:
    #      - The enemy is at that candidate cell, and
    #      - Its energy is higher than the allied agent's.
    #    Shape: (n_envs, 16, 5, 16)
    threat_per_enemy = is_same & enemy_stronger

    #    For each candidate move (over all enemy agents), reduce with OR.
    #    Shape: (n_envs, 16, 5)
    threat = jnp.any(threat_per_enemy, axis=-1)

    # 4. Create the final mask.
    #    The rule is: if any adjacent cell (actions 1-4) is threatened, then both that move 
    #    and the center (stay) action should be masked.
    
    #    First, aggregate threats for the four movement actions (ignoring center at index 0).
    #    Shape: (n_envs, 16)
    adjacent_threat = jnp.any(threat[:, :, 1:], axis=-1)

    #    For the center action: allowed only if no adjacent threat exists.
    #    Shape: (n_envs, 16)
    allowed_center = ~adjacent_threat

    #    For the movement actions (indices 1-4): allowed if that candidate cell is not threatened.
    #    Shape: (n_envs, 16, 4)
    allowed_adjacent = ~threat[:, :, 1:]

    #    Concatenate to form the final mask.
    #    The final_mask shape is (n_envs, 16, 5) corresponding to [center, up, right, down, left]
    final_mask = jnp.concatenate([allowed_center[:, :, None], allowed_adjacent], axis=-1)
    
    return final_mask

all_directions = jnp.array([
    [0, 0],   # center (stay)
    [0, -1],  # move up
    [1, 0],   # move right
    [0, 1],   # move down
    [-1, 0],  # move left
], dtype=jnp.int16)


# @jax.jit
def get_actions(
    rng,
    team_idx: int,
    opponent_idx: int,
    logits,
    observations,
    sap_ranges,
    sap_costs,
    relic_nodes,
    points_map,
):
    n_envs = observations.units.position.shape[0]
    
    agent_positions = observations.units.position[:, team_idx, ..., None, :] 
    agent_positions = agent_positions if team_idx == 0 else transform_coordinates(agent_positions)

    new_positions = agent_positions + directions

    in_bounds = (
        (new_positions[..., 0] >= 0) & (new_positions[..., 0] <= Constants.MAP_WIDTH - 1) &
        (new_positions[..., 1] >= 0) & (new_positions[..., 1] <= Constants.MAP_HEIGHT - 1)
    )

    asteroid_tiles = observations.map_features.tile_type == Constants.ASTEROID_TILE
    asteroid_tiles = asteroid_tiles if team_idx == 0 else transform_observation(asteroid_tiles)

    is_asteroid = asteroid_tiles[
        0, 
        new_positions[..., 0].clip(0, Constants.MAP_WIDTH - 1),
        new_positions[..., 1].clip(0, Constants.MAP_HEIGHT - 1),
    ]
    valid_movements = in_bounds & (~is_asteroid)

    agent_positions_for_collision = observations.units.position[:, team_idx, ...]
    opponent_positions_for_collision = observations.units.position[:, opponent_idx, ...]
    agent_positions_for_collision = agent_positions_for_collision if team_idx == 0 else transform_coordinates(agent_positions_for_collision)
    opponent_positions_for_collision = opponent_positions_for_collision if team_idx == 0 else transform_coordinates(opponent_positions_for_collision)
    valid_collision_avoidance = compute_collision_avoidance(
        agent_positions_for_collision,
        observations.units.energy[:, team_idx, :],
        opponent_positions_for_collision,
        observations.units.energy[:, opponent_idx, :],
        all_directions,
    )

    adjacent_offsets = jnp.array(
        [
            [0, 0],
            [-1, 0],
            [0, -1],
            [0, 1],
            [1, 0],
        ], dtype=jnp.int16
    )

    # TARGET NON-ZERO ENERGY OPPONENTS
    opponent_positions = observations.units.position[:, opponent_idx, ..., None, :] 
    opponent_energy = observations.units.energy[:, opponent_idx, :, None, None].repeat(2, axis=-1)
    opponent_positions = jnp.where(
        opponent_energy > 0,
        opponent_positions,
        -1
    )
    opponent_positions = opponent_positions if team_idx == 0 else transform_coordinates(opponent_positions)

    opponent_positions = jnp.where(
        opponent_positions == -1,
        -100,
        opponent_positions,
    )

    opponent_positions = jnp.where(
        opponent_positions == 24,
        -100,
        opponent_positions,
    )
    opponent_targets = opponent_positions + adjacent_offsets

    # TARGET 5X5 RELIC NODES IN THE DARK
    adjacent_offsets_5x5 = jnp.array(
        [
            [-2, -2], [-2, -1], [-2, 0], [-2, 1], [-2, 2],
            [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
            [ 0, -2], [ 0, -1], [ 0, 0], [ 0, 1], [ 0, 2],
            [ 1, -2], [ 1, -1], [ 1, 0], [ 1, 1], [ 1, 2],
            [ 2, -2], [ 2, -1], [ 2, 0], [ 2, 1], [ 2, 2],
        ], dtype=jnp.int16
    )

    relic_nodes_positions = relic_nodes if team_idx == 1 else transform_coordinates(relic_nodes)
    relic_nodes_positions = jnp.where(
        relic_nodes_positions == -1,
        -100,
        relic_nodes_positions,
    )
    relic_nodes_positions = jnp.where(
        relic_nodes_positions == 24,
        -100,
        relic_nodes_positions,
    )
 
    relic_targets = relic_nodes_positions[:, :, None, :] + adjacent_offsets_5x5
    relic_targets = jnp.where(
        relic_targets > 0,
        relic_targets,
        -1
    )
    relic_targets = jnp.where(
        relic_targets < 24,
        relic_targets,
        -1
    )

    sensor_mask = observations.sensor_mask.transpose(0, 2, 1)
    sensor_mask = sensor_mask if team_idx == 0 else transform_observation(sensor_mask)

    relic_targets_mask = filter_targets_with_boolean_map(
        relic_targets,
        ~sensor_mask
    )
    points_targets_mask = filter_targets_with_boolean_map(
        relic_targets,
        points_map == 1,
    )
    relic_targets_mask = relic_targets_mask & points_targets_mask

    relic_targets = jnp.where(
        relic_targets_mask[..., None].repeat(2, axis=-1),
        relic_targets,
        -100,
    )
    relic_targets = jnp.where(
        relic_targets < 0,
        -100,
        relic_targets,
    )

    target_positions = jnp.concatenate([
        opponent_targets.reshape(n_envs, -1, 2),
        relic_targets.reshape(n_envs, -1, 2),
    ], axis=1)

    target_x, _ = generate_attack_masks_batch(
        agent_positions.reshape(n_envs, -1, 2),
        target_positions,
        sap_ranges,
        sap_ranges
    )
    logits2_mask = target_x

    valid_movements = jnp.concatenate([
        jnp.ones((1, n_envs * 16, 1), dtype=jnp.bool), # this allows center movement
        valid_movements.reshape(1, -1, 4),
        jnp.ones((1, n_envs * 16, 1), dtype=jnp.bool), # this allows sap actions
    ], axis=-1)

    valid_collision_avoidance = jnp.concatenate([
        valid_collision_avoidance,
        jnp.expand_dims(valid_collision_avoidance.sum(axis=-1) == 5, axis=-1) # if you have to run, don't sap
    ], axis=-1)
    valid_movements = valid_movements & valid_collision_avoidance.reshape(1, -1, 6)

    sap_mask = jnp.concatenate([
        jnp.ones((1, n_envs * 16, 5)), # allow all movements
        target_x.sum(axis=-1).reshape(1, n_envs * 16, 1),
    ], axis=-1)

    non_negative_energy_mask = jnp.concatenate([
        jnp.ones((1, n_envs * 16, 1), dtype=jnp.bool), # allow only NO-OP
        (observations.units.energy[:, team_idx, :].reshape(-1) > 0)[None, :, None].repeat(5, axis=-1),
    ], axis=-1)
    
    logits1_mask = valid_movements & (sap_mask > 0) & non_negative_energy_mask

    logits1, logits2, logits3 = logits

    logits1_mask = logits1_mask.reshape(logits1.shape)
    logits2_mask = logits2_mask.reshape(logits2.shape)
    large_negative = -1e9
    masked_logits1 = jnp.where(logits1_mask.reshape(logits1.shape), logits1, large_negative)
    masked_logits2 = jnp.where(logits2_mask.reshape(logits2.shape), logits2, large_negative)

    dist1 = distrax.Categorical(logits=masked_logits1)
    dist2 = distrax.Categorical(logits=masked_logits2)

    rng, action_rng1, action_rng2, action_rng3 = jax.random.split(rng, num=4)
    actions1, log_probs1 = dist1.sample_and_log_prob(seed=action_rng1)
    actions2, log_probs2 = dist2.sample_and_log_prob(seed=action_rng2)

    target_y = jax.vmap(
        generate_attack_masks_y,
        in_axes=(0, 0, 0, 0, 0)
    )(
        agent_positions.reshape(n_envs, -1, 2),
        target_positions.reshape(n_envs, -1, 2),
        sap_ranges,
        sap_ranges,
        actions2.reshape(n_envs, -1) - Constants.MAX_SAP_RANGE
    )

    logits3_mask = target_y
    logits3_mask = logits3_mask.reshape(logits3.shape)
    masked_logits3 = jnp.where(logits3_mask.reshape(logits3.shape), logits3, large_negative)

    dist3 = distrax.Categorical(logits=masked_logits3)

    actions3, log_probs3 = dist3.sample_and_log_prob(seed=action_rng3)
    actions = [actions1, actions2, actions3]

    target_log_probs_mask = (actions1 == 5)
    log_probs = log_probs1 + jnp.where(target_log_probs_mask, log_probs2, 0) + jnp.where(target_log_probs_mask, log_probs3, 0)

    actions = jnp.stack([actions1.reshape(-1), actions2.reshape(-1), actions3.reshape(-1)])
    actions = actions.T.reshape(n_envs, 16, -1)

    logits_mask = [
        logits1_mask,
        logits2_mask,
        logits3_mask
    ]

    return actions, log_probs, logits_mask

def generate_attack_masks(agent_positions, target_positions, x_range=8, y_range=8, choose_y=False, chosen_x=None,):
    """
    Generate attack masks for agents based on both x and y distances to targets.
    Targets outside the range are filtered out before mask generation.
    
    Args:
        agent_positions (jnp.ndarray): Shape (num_agents, 2) array of agent positions
        target_positions (jnp.ndarray): Shape (num_targets, 2) array of target positions
        x_range (int): Maximum x-distance range (default 8)
        y_range (int): Maximum y-distance range (default 8)
    
    Returns:
        attack_masks (jnp.ndarray): Shape (num_agents, 17, 17) boolean array
                                   True indicates valid attack position at that offset
    """
    # Pre-filter invalid targets (marked as -1)
    valid_targets = target_positions != -1
    valid_targets = jnp.all(valid_targets, axis=-1)
    target_positions = jnp.where(valid_targets[:, None], target_positions, 1000)
    
    # Calculate x and y distances from agent to each target
    x_distances = target_positions[None, :, 0] - agent_positions[:, None, 0]
    y_distances = target_positions[None, :, 1] - agent_positions[:, None, 1]
    
    # Create range mask for targets
    targets_in_range = (jnp.abs(x_distances) <= x_range) & (jnp.abs(y_distances) <= y_range)
    targets_in_range = targets_in_range & valid_targets[None, :]

    target_positions = jnp.where(target_positions == -1, 1000, target_positions)

    x_distances = jnp.where(
        targets_in_range,
        x_distances,
        -100, 
    )
    y_distances = jnp.where(
        targets_in_range,
        y_distances,
        -100,
    )
    
    x_offsets = jnp.arange(-8, 9)
    y_offsets = jnp.arange(-8, 9)
    
    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Check valid positions for x and y separately
    valid_x = (x_distances == x_offsets)
    valid_x = jnp.any(valid_x, axis=-1)

    if choose_y:
        x_distances = x_distances[:, None, :]
        y_distances = y_distances[:, None, :]
        x_offsets = x_offsets[None, :, None]
        y_offsets = y_offsets[None, :, None]
        
        # Filter targets based on chosen x
        chosen_x = chosen_x[:, None, None]  # Shape: (num_agents, 1, 1)
        valid_targets_for_x = (x_distances == chosen_x)
        
        # Apply the x-based filter to y distances
        y_distances = jnp.where(valid_targets_for_x, y_distances, -100)
        
        # Generate y masks only for valid targets based on chosen x
        valid_y = (y_distances == y_offsets)
        valid_y = jnp.any(valid_y, axis=-1)

        indices = jnp.arange(valid_y.shape[1])

        # Use advanced indexing to extract the desired slices
        final_filter = valid_y[0, indices, indices, :]
        return final_filter

    valid_y = (y_distances == y_offsets)
    valid_y = jnp.any(valid_y, axis=-1)

    return valid_x, valid_y


generate_attack_masks_batch = jax.vmap(
    generate_attack_masks,
    in_axes=(0, 0, 0, 0)
)

def generate_attack_masks_y(agent_positions, target_positions, x_range=8, y_range=8, chosen_x=None):
    """
    Generate attack masks for agents based on both x and y distances to targets.
    Targets outside the range are filtered out before mask generation.
    
    Args:
        agent_positions (jnp.ndarray): Shape (num_agents, 2) array of agent positions
        target_positions (jnp.ndarray): Shape (num_targets, 2) array of target positions
        x_range (int): Maximum x-distance range (default 8)
        y_range (int): Maximum y-distance range (default 8)
    
    Returns:
        attack_masks (jnp.ndarray): Shape (num_agents, 17, 17) boolean array
                                   True indicates valid attack position at that offset
    """
    # Pre-filter invalid targets (marked as -1)
    valid_targets = target_positions != -1
    valid_targets = jnp.all(valid_targets, axis=-1)
    target_positions = jnp.where(valid_targets[:, None], target_positions, 1000)
    
    # Calculate x and y distances from agent to each target
    x_distances = target_positions[None, :, 0] - agent_positions[:, None, 0]
    y_distances = target_positions[None, :, 1] - agent_positions[:, None, 1]
    
    # Create range mask for targets
    targets_in_range = (jnp.abs(x_distances) <= x_range) & (jnp.abs(y_distances) <= y_range)
    targets_in_range = targets_in_range & valid_targets[None, :]

    target_positions = jnp.where(target_positions == -1, 1000, target_positions)

    x_distances = jnp.where(
        targets_in_range,
        x_distances,
        -100, 
    )
    y_distances = jnp.where(
        targets_in_range,
        y_distances,
        -100,
    )
    
    x_offsets = jnp.arange(-8, 9)
    y_offsets = jnp.arange(-8, 9)
    
    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Check valid positions for x and y separately
    valid_x = (x_distances == x_offsets)
    valid_x = jnp.any(valid_x, axis=-1)

    x_distances = x_distances[:, None, :]
    y_distances = y_distances[:, None, :]
    x_offsets = x_offsets[None, :, None]
    y_offsets = y_offsets[None, :, None]
    
    # Filter targets based on chosen x
    chosen_x = chosen_x[:, None, None]  # Shape: (num_agents, 1, 1)
    valid_targets_for_x = (x_distances == chosen_x)
    
    # Apply the x-based filter to y distances
    y_distances = jnp.where(valid_targets_for_x, y_distances, -100)
    
    # Generate y masks only for valid targets based on chosen x
    valid_y = (y_distances == y_offsets)
    valid_y = jnp.any(valid_y, axis=-1)

    indices = jnp.arange(valid_y.shape[1])

    # Use advanced indexing to extract the desired slices
    final_filter = valid_y[0, indices, indices, :]
    return final_filter
