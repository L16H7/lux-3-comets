import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import NamedTuple


class Transition(NamedTuple):
    episode_info: jnp.ndarray
    states: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    units_mask: jnp.ndarray
    agent_positions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    logits1_mask: jnp.ndarray
    logits2_mask: jnp.ndarray
    logits3_mask: jnp.ndarray
    env_information: jnp.ndarray


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition: Transition):
        values = transition.values
        gae, next_value = gae_and_next_value
        delta = transition.rewards + gamma * next_value * (1 - transition.dones) - values
        gae = delta + gamma * gae_lambda * (1 - transition.dones) * gae
        return (gae, values), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.values

def ppo_update(
    actor_train_state: TrainState,
    transitions: Transition,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    units_mask = transitions.units_mask
    active_units = units_mask.sum() + 1e-8

    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)

    def _loss_fn(actor_params):
        logits, values = actor_train_state.apply_fn(
            actor_params,
            {
                "states": transitions.states,
                "positions": transitions.agent_positions,
                "match_steps": transitions.episode_info[:, 0],
                "matches": transitions.episode_info[:, 1],
                "team_points": transitions.episode_info[:, 2],
                "opponent_points": transitions.episode_info[:, 3],
                "unit_move_cost": transitions.env_information[:, 0],
                "unit_sap_cost": transitions.env_information[:, 1],
                "unit_sap_range": transitions.env_information[:, 2],
                "unit_sensor_range": transitions.env_information[:, 3],
            }
        )
        logits1, logits2, logits3 = logits
        
        large_negative = -1e9
        masked_logits1 = jnp.where(
            transitions.logits1_mask,
            logits1,
            large_negative,
        )

        masked_logits2 = jnp.where(
            transitions.logits2_mask,
            logits2,
            large_negative,
        )

        masked_logits3 = jnp.where(
            transitions.logits3_mask,
            logits3,
            large_negative
        )

        dist1 = distrax.Categorical(logits=masked_logits1)
        dist2 = distrax.Categorical(logits=masked_logits2)
        dist3 = distrax.Categorical(logits=masked_logits3)

        log_probs1 = dist1.log_prob(transitions.actions[..., 0])
        log_probs2 = dist2.log_prob(transitions.actions[..., 1])
        log_probs3 = dist3.log_prob(transitions.actions[..., 2])

        target_log_probs_mask = (transitions.actions[..., 0] == 5)
        log_probs = log_probs1 + jnp.where(target_log_probs_mask, log_probs2, 0) + jnp.where(target_log_probs_mask, log_probs3, 0)

        log_ratio = log_probs - transitions.log_probs
        log_ratio = jnp.where(units_mask, log_ratio, 0)
        log_ratio = jnp.where(
            units_mask.sum(axis=-1) > 0, 
            log_ratio.sum(axis=-1) / units_mask.sum(axis=-1), 
            0
        )
       
        ratio = jnp.exp(log_ratio)

        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).sum() / active_units
        
        entropy1 = jnp.where(units_mask, dist1.entropy(), 0)
        target_log_probs_mask = jnp.where(units_mask, target_log_probs_mask, 0)
        entropy2 = jnp.where(target_log_probs_mask, dist2.entropy(), 0)
        entropy3 = jnp.where(target_log_probs_mask, dist3.entropy(), 0)

        entropy = entropy1 + entropy2 + entropy3

        entropy_loss = entropy.sum() / active_units

        values = jnp.squeeze(values)

        value_pred_clipped = transitions.values + (values - transitions.values).clip(-clip_eps, clip_eps)

        value_loss = jnp.square(values - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        loss = actor_loss + vf_coef * value_loss - ent_coef * entropy_loss

        explained_var = 1 - jnp.var(values - targets) / (jnp.var(targets) + 1e-8)
        approx_kl = ((ratio - 1.0) - log_ratio).sum() / active_units # http://joschu.net/blog/kl-approx.html
        clip_frac = (abs((ratio - 1.0)) > clip_eps).sum() / active_units

        update_info = {
            "explained_var": explained_var,
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "value_mean": jnp.mean(transitions.values),
            "value_std": jnp.std(transitions.values),
            "value_loss": value_loss,
            "actor_loss": actor_loss,
            "entropy": entropy_loss,
        }

        return loss, update_info

    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
    (loss, update_info), grads = grad_fn(actor_train_state.params)
    (loss, update_info) = jax.lax.pmean((loss, update_info), axis_name="devices")

    def mean_leaf(x):
        return jnp.mean(x)
    
    def std_leaf(x):
        return jnp.std(x)

    grads_mean = jax.tree_util.tree_map(mean_leaf, grads)
    grads_std = jax.tree_util.tree_map(std_leaf, grads)

    updated_actor_train_state = actor_train_state.apply_gradients(grads=grads)

    update_step_info = {
        **update_info,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "loss": loss,
        "actor_dense6_mean": grads_mean['params']['Dense_6']['kernel'],
        "actor_dense6_std": grads_std['params']['Dense_6']['kernel'],
    }
    return updated_actor_train_state, update_step_info 
