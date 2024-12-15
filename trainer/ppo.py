import distrax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from typing import NamedTuple


class Transition(NamedTuple):
    observations: jnp.ndarray
    episode_info: jnp.ndarray
    agent_episode_info: jnp.ndarray
    states: jnp.ndarray
    prev_actions: jnp.ndarray
    prev_rewards: jnp.ndarray
    actions: jnp.ndarray
    log_probs: jnp.ndarray
    values: jnp.ndarray
    units_mask: jnp.ndarray
    agent_positions: jnp.ndarray
    team_positions: jnp.ndarray
    opponent_positions: jnp.ndarray
    relic_nodes_positions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition: Transition):
        values = transition.values.repeat(16)
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
    return advantages, advantages + transitions.values.repeat(16, axis=1)

def ppo_update(
    actor_train_state: TrainState,
    critic_train_state: TrainState,
    transitions: Transition,
    actor_hstates: jax.Array,
    critic_hstates: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    units_mask = transitions.units_mask.reshape(-1)
    active_units = units_mask.sum() + 1e-8

    advantages = advantages.reshape(-1)
    adv_mean = advantages.mean()
    adv_std = advantages.std()
    advantages = (advantages - adv_mean) / (adv_std + 1e-8)
    advantages = advantages * units_mask

    def _loss_fn(actor_params, critic_params):
        logits, _ = actor_train_state.apply_fn(
            actor_params,
            actor_hstates,
            {
                "observations": transitions.observations,
                "prev_actions": transitions.prev_actions,
                "positions": transitions.agent_positions,
                "relic_nodes_positions": transitions.relic_nodes_positions,
                "team_positions": transitions.team_positions,
                "opponent_positions": transitions.opponent_positions,
                "prev_rewards": jnp.expand_dims(transitions.prev_rewards, axis=2),
                "teams": transitions.agent_episode_info[:, :, 0].astype(jnp.int32),
                "match_phases": transitions.agent_episode_info[:, :, 1].astype(jnp.int32),
                "matches": transitions.agent_episode_info[:, :, 2].astype(jnp.int32),
                "team_points": jnp.expand_dims(transitions.agent_episode_info[:, :, 3], axis=2),
                "opponent_points": jnp.expand_dims(transitions.agent_episode_info[:, :, 4], axis=2),
            }
        )
        logits1, logits2, logits3 = logits
        dist1 = distrax.Categorical(logits=logits1)
        dist2 = distrax.Categorical(logits=logits2)
        dist3 = distrax.Categorical(logits=logits3)
        dist = distrax.Joint([dist1, dist2, dist3])

        n_steps, n_agents = transitions.observations.shape[:2]
        log_probs = dist.log_prob(
            [
                transitions.actions[:, :, :, 0].reshape(n_steps, 1, n_agents),
                transitions.actions[:, :, :, 1].reshape(n_steps, 1, n_agents),
                transitions.actions[:, :, :, 2].reshape(n_steps, 1, n_agents)
            ]
        )
        log_ratio = log_probs.reshape(-1) - transitions.log_probs.reshape(-1)
        log_ratio = log_ratio * units_mask
        
        ratio = jnp.exp(log_ratio)

        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).sum() / active_units
        
        entropy = dist.entropy().reshape(-1) * units_mask
        entropy_loss = entropy.sum() / active_units

        values, _ = critic_train_state.apply_fn(
            critic_params,
            critic_hstates,
            {
                "states": transitions.states,
                "teams": transitions.episode_info[:, :, 0].astype(jnp.int32),
                "match_phases": transitions.episode_info[:, :, 1].astype(jnp.int32),
                "matches": transitions.episode_info[:, :, 2].astype(jnp.int32),
                "team_points": jnp.expand_dims(transitions.episode_info[:, :, 3], axis=2),
                "opponent_points": jnp.expand_dims(transitions.episode_info[:, :, 4], axis=2),
            }
        )
        values = values.reshape(-1)

        value_pred_clipped = transitions.values.reshape(-1) + (values - transitions.values.reshape(-1)).clip(-clip_eps, clip_eps)

        # this is needed because we calculate targets for each agent in a state
        value_targets = targets.reshape(targets.shape[0], -1, 16).mean(axis=2).reshape(-1)
        value_loss = jnp.square(values - value_targets)
        value_loss_clipped = jnp.square(value_pred_clipped - value_targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        loss = actor_loss + vf_coef * value_loss - ent_coef * entropy_loss

        explained_var = 1 - jnp.var(values - value_targets) / (jnp.var(value_targets) + 1e-8)

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

    grad_fn = jax.value_and_grad(_loss_fn, argnums=(0, 1), has_aux=True)
    (loss, update_info), grads = grad_fn(
        actor_train_state.params, critic_train_state.params
    )
    actor_grads, critic_grads = grads
    updated_actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
    updated_critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

    update_step_info = {
        **update_info,
        "adv_mean": adv_mean,
        "adv_std": adv_std,
        "loss": loss,
    }
    return updated_actor_train_state, updated_critic_train_state, update_step_info 
