import jax
import jax.numpy as jnp


def calculate_nebula_map(
    sensor_map,
    nebula_map,
    prev_agent_energies,
    agent_positions,
    agent_energies,
    agent_mask,
    energy_map,  
):
    # mask of agents on nebula
    # nebula energy_deduction = agent_energies - prev_agent_energies - energy_map 
    # nebular energy deduction [mask] 
    # sum() / mask.sum()
    # update nebula energy deduction 
    agent_x = agent_positions[..., 0]
    agent_y = agent_positions[..., 1]

    non_base_agents_x = (agent_x > 0) & (agent_x < 23)
    non_base_agents_y = (agent_y > 0) & (agent_y < 23)
    non_base_agents = non_base_agents_x & non_base_agents_y
    
    # Create the mask by indexing nebula_map with agent positions
    nebula_agent_mask = nebula_map[jnp.arange(nebula_map.shape[0])[:, None], agent_y, agent_x]
    nebula_agent_mask = nebula_agent_mask & non_base_agents

    energy_for_agents = energy_map[jnp.arange(energy_map.shape[0])[:, None], agent_y, agent_x]

    nebula_energy_deduction = agent_energies - prev_agent_energies - energy_for_agents
    nebula_energy_deduction = nebula_energy_deduction * nebula_agent_mask * agent_mask
    nebula_energy_deduction = nebula_energy_deduction.sum(axis=-1) * nebula_agent_mask.sum(axis=-1)
    nebula_energy_deduction_correction_mask = (nebula_energy_deduction >= -25) & (nebula_energy_deduction <= 0)

    jax.debug.breakpoint()
    return
