import jax
import jax.numpy as jnp


def calculate_sapping_stats(
    actions,
    units_position,
    units_mask,
    opponent_units_position,
    opponent_units_mask,
):
    sap_actions_mask = actions[:, 0] == 5
    sap_actions_mask = sap_actions_mask * units_mask
    units_mask = jnp.expand_dims(units_mask, axis=-1)
    opponent_units_mask = jnp.expand_dims(opponent_units_mask, axis=-1)

    opponent_units_position = jnp.where(~opponent_units_mask.repeat(2, axis=-1), jnp.inf, opponent_units_position)

    sap_actions_delta = actions[:, 1:] * jnp.expand_dims(sap_actions_mask, axis=-1)
    sapped_targets = units_position + sap_actions_delta
    sapped_targets = jnp.where(~units_mask.repeat(2, axis=-1), -jnp.inf, sapped_targets)

    sapped_targets_expanded = sapped_targets[:, None, :]
    opponent_units_position_expanded = opponent_units_position[None, :, :]

    # # Element-wise comparison and reducing to check if any target matches each shot
    direct_hits = jnp.all(sapped_targets_expanded == opponent_units_position_expanded, axis=2)  # Shape (10, 5)
    direct_hits_per_opponent = jnp.sum(direct_hits, axis=0)
    total_direct_hits = jnp.sum(direct_hits_per_opponent)

    adjacent_offsets = jnp.array(
        [
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1],
        ], dtype=jnp.int16
    )
    indirect_sapped_targets = sapped_targets[:, None, :] + adjacent_offsets
    indirect_sapped_targets = indirect_sapped_targets.reshape(-1, 2)
    indirect_sapped_targets_expanded = indirect_sapped_targets[:, None, :]

    indirect_hits = jnp.all(indirect_sapped_targets_expanded == opponent_units_position_expanded, axis=2)
    indirect_hits_per_opponent = jnp.sum(indirect_hits, axis=0)
    total_indirect_hits = jnp.sum(indirect_hits_per_opponent)

    return {
        "total_direct_hits": total_direct_hits,
        "total_indirect_hits": total_indirect_hits,
        "total_sapped_actions": sap_actions_mask.sum(),
    }
