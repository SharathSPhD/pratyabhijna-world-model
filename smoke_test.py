import sys
import logging
sys.path.insert(0, '/home/sharaths/projects/pwm-phase1')

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(name)s: %(message)s',
)

from omegaconf import OmegaConf
from pwm.scripts.train import PWMTrainer

cfg = OmegaConf.create({
    "world_model": {
        "levels": 1, "stoch_dim": 32, "stoch_classes": 32,
        "hidden_dim_apara": 512, "backbone_apara": "gru",
        "kl_balance_dyn": 0.5, "kl_balance_rep": 0.1,
        "free_bits": 1.0, "obs_dim": 256, "action_dim": 64,
        "vocab_size": 32000,
    },
    "training": {
        "batch_size": 4, "seq_len": 16, "lr_wm": 1e-4,
        "lr_actor": 3e-5, "lr_critic": 1e-4,
        "gradient_clip": 100.0, "replay_capacity": 1000,
        "min_buffer_steps": 50, "mixed_precision": "bfloat16",
        "torch_compile": False, "phase_a_steps": 50,
        "phase_b_steps": 5, "phase_c_steps": 5,
        "eval_interval": 100, "checkpoint_interval": 1000, "seed": 42,
    },
    "corpus": {
        # Point to the main worktree corpus which has 4936 .txt files
        "data_dir": "/home/sharaths/projects/PWM/data/corpus",
        "window_size": 64, "stride": 16, "max_tokens": 10000,
    },
    "sphuratta": {"percentile": 5, "min_gap": 50},
    "logging": {"wandb_project": "pwm-smoke", "log_interval": 10},
    "llm": {"enabled": False},
    "memory": {"enabled": False},
    "sleep": {"enabled": False},
    "actor": {"type": "reinforce", "horizon": 5, "gamma": 0.99, "lam": 0.95, "entropy_coef": 3e-4},
    "reward": {"alpha_1": 1.0, "alpha_2": 0.0, "alpha_3": 0.0, "lambda_ext": 0.0},
})

print("=== Constructing PWMTrainer ===")
trainer = PWMTrainer(cfg)
print(f"=== Env type: {type(trainer.env).__name__} ===")
print("=== Starting 50-step smoke run ===")
metrics = trainer.train(n_steps=50)
print(f"SMOKE DONE: {metrics}")
