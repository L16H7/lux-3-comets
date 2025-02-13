import jax
import jax.numpy as jnp

from utils import transform_observation_3dim


def calculate_nebula_map(
    sensor_map,
    sensor_range,
    nebula_map,
    nebula_info,
    prev_agent_positions,
    prev_agent_energies,
    agent_positions,
    agent_energies,
    agent_mask,
    energy_map,  
):
    # mask of agents on nebula
    # nebula energy_reduction = agent_energies - prev_agent_energies - energy_map 
    # nebular energy reduction [mask] 
    # sum() / mask.sum()
    # update nebula energy reduction 
    agent_x = agent_positions[..., 0]
    agent_y = agent_positions[..., 1]

    non_base_agents_x = (agent_x > 0) & (agent_x < 23)
    non_base_agents_y = (agent_y > 0) & (agent_y < 23)
    non_base_agents = non_base_agents_x | non_base_agents_y
    sufficient_energy_agents = (prev_agent_energies > 25) & (agent_energies > 0)
    moving_agents = (prev_agent_positions != agent_positions).sum(axis=-1) == 1
    
    # Create the mask by indexing nebula_map with agent positions
    nebula_agent_mask = nebula_map[jnp.arange(nebula_map.shape[0])[:, None], agent_y, agent_x]
    nebula_agent_mask = nebula_agent_mask & non_base_agents & sufficient_energy_agents & agent_mask & moving_agents

    energy_for_agents = energy_map[jnp.arange(energy_map.shape[0])[:, None], agent_y, agent_x]

    nebula_energy_reduction = agent_energies - prev_agent_energies - energy_for_agents
    nebula_energy_reduction = nebula_energy_reduction * nebula_agent_mask

    nebula_energy_reduction = nebula_energy_reduction.sum(axis=-1) / nebula_agent_mask.sum(axis=-1)

    allowed_values = jnp.array([-25, -5, -4, -3, -2, -1, 0])

    # Create a mask
    nebula_energy_reduction_correction_mask = jnp.isin(nebula_energy_reduction, allowed_values)
    nebula_update_mask = (nebula_agent_mask.sum(axis=-1) > 0) & nebula_energy_reduction_correction_mask

    nebula_info_not_updated = (nebula_info[:, 1] == 0)
    nebula_update_mask = nebula_update_mask & nebula_info_not_updated

    updated_nebula_info = jnp.where(
        nebula_update_mask[:, None].repeat(2, axis=-1),
        jnp.concatenate([
            nebula_energy_reduction[:, None],
            jnp.ones_like(nebula_energy_reduction[:, None])
        ], axis=-1),
        nebula_info
    )

    updated_nebula_energy_reduction = updated_nebula_info[:, 0]

    calculated_sensor_map = construct_calculated_sensor_map(
        agent_positions,
        sensor_range,
    )
    augmented_nebula_map = calculated_sensor_map & ~sensor_map
    transformed_augmented_nebula_map = transform_observation_3dim(augmented_nebula_map)
    augmented_nebula_map = jnp.where(
        transformed_augmented_nebula_map != 0,
        transformed_augmented_nebula_map,
        augmented_nebula_map,
    )

    nebula_map = jnp.where(
        augmented_nebula_map != 0,
        augmented_nebula_map,
        nebula_map,
    )
    updated_nebula_map = nebula_map * ((updated_nebula_energy_reduction - 1) * 0.05)[:, None, None]

    return updated_nebula_info, updated_nebula_map


def construct_calculated_sensor_map(unit_positions, unit_sensor_range, map_size=24):
    n_envs, n_units, _ = unit_positions.shape
    
    # Mask invalid unit positions
    valid_mask = jnp.any(unit_positions != -1, axis=-1)
    
    # Calculate sensor grid bounds
    x_min = jnp.maximum(0, unit_positions[:, :, 0] - unit_sensor_range[:, None])
    x_max = jnp.minimum(map_size, unit_positions[:, :, 0] + unit_sensor_range[:, None] + 1)
    y_min = jnp.maximum(0, unit_positions[:, :, 1] - unit_sensor_range[:, None])
    y_max = jnp.minimum(map_size, unit_positions[:, :, 1] + unit_sensor_range[:, None] + 1)
    
    # Replace bounds for invalid units with invalid values
    x_min = jnp.where(valid_mask, x_min, map_size)
    x_max = jnp.where(valid_mask, x_max, 0)
    y_min = jnp.where(valid_mask, y_min, map_size)
    y_max = jnp.where(valid_mask, y_max, 0)
    
    # Create indices for visible areas
    x_idx = jnp.arange(map_size)[None, None, :]
    y_idx = jnp.arange(map_size)[None, None, :]
    
    # Mask visible areas
    visible_mask_x = (x_idx >= x_min[:, :, None]) & (x_idx < x_max[:, :, None])
    visible_mask_y = (y_idx >= y_min[:, :, None]) & (y_idx < y_max[:, :, None])
    
    # Combine masks
    visible_mask = jnp.any((visible_mask_x[:, :, None, :] & visible_mask_y[:, :, :, None]), axis=1)
    
    return visible_mask
