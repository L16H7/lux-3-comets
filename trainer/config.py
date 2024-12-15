from dataclasses import dataclass


@dataclass
class Config:
    n_meta_steps: int = 1
    n_actor_steps: int = 16
    n_update_steps: int = 32
    n_envs_per_device: int = 2048
    n_envs: int = 2048
    n_eval_envs: int = 512
    n_minibatches: int = 8
    train_seed: int = 42
    map_width: int = 24
    map_height: int = 24
    wandb_project: str = "multi-self-play-test"
    checkpoint_path: str ="./checkpoints"
    wandb_api_key: str = None
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 5e-4
    batch_size: int = 4096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    policy_clip: float = 0.20
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    n_epochs: int = 1
    max_steps_in_match: int = 100
    n_agents: int = 16
