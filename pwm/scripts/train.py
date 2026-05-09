"""
PWMTrainer: Three-phase DreamerV3 training loop for the Pratyabhijñā World Model.

Philosophical grounding:
  Pañcakṛtya (MV 1.4, Kṣemarāja; ĪPK 3.1–3.2, Utpaladeva): The five divine acts
  of Śiva are mirrored in the three training phases. Phase A (cit/ābhāsana) trains
  the world model to manifest the generative process. Phase B (icchā) trains the
  EFE actor — the will that selects from imagined futures. Phase C (jñāna/vimarśa)
  trains the critic — the evaluative reflection that assigns value.

  Svātantrya (ĪPK 2.1, Utpaladeva): The maximum-entropy policy regulariser in
  Phase B preserves the system's creative freedom — the policy must not collapse
  to a deterministic output, lest it lose the capacity for genuine spanda.

  Camatkāra (Locana ad DhvA 1.1, Abhinavagupta): During Phases B and C, the
  standard extrinsic reward is replaced by R_camatk — the intrinsic wonder signal.
  This operationalises the claim that creative value is self-certified, not imposed.

Three-phase loop (DreamerV3 protocol, validated on DGX Spark GB10):
  A — World Model: Adam lr=1e-4, grad_clip=1000, B=32, T=64
  B — Actor:       Adam lr=3e-5, grad_clip=100,  horizon H=13, λ-returns γ=0.99, λ=0.95
  C — Critic:      twohot CE + EMA slow critic (decay=0.98), grad_clip=100
"""

from __future__ import annotations

import os
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Categorical
from dotenv import load_dotenv  # type: ignore[import]

load_dotenv()

try:
    import hydra  # type: ignore[import]
    from omegaconf import DictConfig, OmegaConf  # type: ignore[import]
    _HYDRA = True
except ImportError:
    _HYDRA = False

try:
    import wandb  # type: ignore[import]
    _WANDB = True
except ImportError:
    _WANDB = False

try:
    import mlflow  # type: ignore[import]
    _MLFLOW = True
except ImportError:
    _MLFLOW = False

from pwm.world_model.trika import TrikaWorldModel  # type: ignore[import]
from pwm.world_model.losses import (  # type: ignore[import]
    make_twohot_bins,
    twohot_encode,
    twohot_loss,
)
from pwm.memory.citta_store import CittaStore  # type: ignore[import]
from pwm.memory.replay import ReplayBuffer, Transition  # type: ignore[import]
from pwm.rewards.camatk import CamatkaraReward  # type: ignore[import]
from pwm.pipeline.pancakrtya_loop import PancakrtyaLoop, LoopConfig  # type: ignore[import]
from pwm.active_inference.efe_actor import EFEActor  # type: ignore[import]   # Phase 2+
from pwm.active_inference.crspp import CRSPPModel    # type: ignore[import]   # Phase 2+
from pwm.data.corpus_dataset import PhaseOneEnv      # type: ignore[import]   # Phase 1+

log = logging.getLogger(__name__)


# ── Phase 0 environment stub ──────────────────────────────────────────────────

class TextEnv:
    """
    Stub environment for Phase 0 / Phase 1 smoke-testing.

    Generates random observation tensors simulating a BPE-embedded text sequence
    (shape: (batch, obs_dim=512)). The obs_dim=512 exceeds the config obs_dim=256
    to ensure the encoder has headroom; the WM encoder projects down internally.

    Replace with the real corpus-driven DataLoader in Phase 1 when the BPE
    tokeniser and GRETIL corpus pipeline are ready.
    """

    def __init__(
        self,
        batch_size: int = 32,
        obs_dim: int = 512,
        action_dim: int = 64,
        seq_len: int = 64,
        device: torch.device | None = None,
    ) -> None:
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.seq_len = seq_len
        self.device = device or torch.device("cpu")
        self._step = 0

    def reset(self) -> Tensor:
        self._step = 0
        return torch.randn(self.batch_size, self.obs_dim, device=self.device)

    def step(self, action: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        del action  # stub: real corpus env would use this for conditioning
        """
        Returns: (obs, reward, done)
        reward is zero — camatkāra reward replaces it during actor/critic training.
        """
        self._step += 1
        obs = torch.randn(self.batch_size, self.obs_dim, device=self.device)
        reward = torch.zeros(self.batch_size, device=self.device)
        done = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        return obs, reward, done

    def sample_batch(self) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Sample a full (B, T, D) trajectory batch for WM sequence training.
        Returns: obs_seq, action_seq, reward_seq, done_seq
        """
        obs_seq = torch.randn(
            self.batch_size, self.seq_len, self.obs_dim, device=self.device
        )
        action_seq = torch.randn(
            self.batch_size, self.seq_len, self.action_dim, device=self.device
        )
        reward_seq = torch.zeros(self.batch_size, self.seq_len, device=self.device)
        done_seq = torch.zeros(
            self.batch_size, self.seq_len, dtype=torch.bool, device=self.device
        )
        return obs_seq, action_seq, reward_seq, done_seq


# ── Critic head ───────────────────────────────────────────────────────────────

class CriticHead(nn.Module):
    """
    Linear distributional critic head on top of WM features.

    Distributional output (twohot CE) rather than scalar regression prevents
    gradient interference when the reward scale shifts at sphurattā events.
    Phase 1 uses a shallow linear critic; Phase 2+ will use an MLP.
    """

    N_BINS = 255
    BINS_LO = -20.0
    BINS_HI = 20.0

    def __init__(self, feature_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(feature_dim, self.N_BINS)
        self.bins: Tensor
        self.register_buffer("bins", make_twohot_bins(self.N_BINS, self.BINS_LO, self.BINS_HI))

    def forward(self, features: Tensor) -> Tensor:
        """Returns logits over N_BINS bins: (B, N_BINS)."""
        return self.net(features)

    def value(self, features: Tensor) -> Tensor:
        """Expected value (scalar per element): (B,)."""
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        return (probs * self.bins).sum(-1)  # type: ignore[operator]


# ── EFE Actor stub (Phase 1: wired but inactive) ──────────────────────────────

class EFEActorStub(nn.Module):
    """
    Placeholder EFE actor for Phase 1.

    Wired into the optimizer and gradient flow now so that Phase 2 can activate
    it by setting `efe_enabled=True` without any architectural changes.
    The stub outputs a zero action and contributes zero gradient.
    """

    def __init__(self, feature_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(feature_dim, action_dim)
        nn.init.zeros_(self.net.weight)
        nn.init.zeros_(self.net.bias)

    def forward(self, features: Tensor) -> Tensor:
        return self.net(features)


# ── Lambda returns ────────────────────────────────────────────────────────────

def _compute_lambda_returns(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float,
    lam: float,
) -> Tensor:
    """
    Compute λ-returns for actor training (Schulman et al. 2015 / DreamerV3 §3).

    G_t^λ = r_t + γ[(1-λ)V(s_{t+1}) + λ G_{t+1}^λ]

    Bootstraps with the critic at the horizon boundary. dones zero out
    future-step contributions, preventing gradient leakage across episodes.
    """
    T = rewards.shape[1]
    targets = torch.zeros_like(rewards)
    last = values[:, -1]
    for t in reversed(range(T)):
        not_done = (~dones[:, t]).float()
        bootstrap = (1.0 - lam) * values[:, t] + lam * last
        targets[:, t] = rewards[:, t] + gamma * not_done * bootstrap
        last = targets[:, t]
    return targets


def _ema_update(target: nn.Module, source: nn.Module, decay: float) -> None:
    """
    EMA parameter update for the slow critic (DreamerV3 §3.3).

    `target = decay * target + (1 - decay) * source`

    The slow critic is used as a value target during actor training to reduce
    gradient variance. EMA decay=0.98 matches the DreamerV3 hyperparameter.
    """
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(decay).add_((1.0 - decay) * sp.data)


# ── PWMTrainer ────────────────────────────────────────────────────────────────

class PWMTrainer:
    """
    Pratyabhijñā World Model trainer — three-phase DreamerV3 loop.

    Instantiation:
        trainer = PWMTrainer(cfg)
        trainer.train(n_steps=500_000)

    The three phases execute within every call to `train_step`:
      Phase A: WM observes a replay batch and minimises VFE (KL + recon).
      Phase B: Actor imagines a horizon from the current WM state and minimises
               the negative λ-return under the camatkāra reward.
      Phase C: Critic minimises twohot CE against λ-returns; slow-critic updated by EMA.
    """

    # DreamerV3 validated hyperparameters (DGX Spark GB10)
    _WM_LR: float = 1e-4
    _WM_WD: float = 1e-6
    _WM_GRAD_CLIP: float = 1000.0

    _ACTOR_LR: float = 3e-5
    _ACTOR_GRAD_CLIP: float = 100.0

    _CRITIC_LR: float = 3e-5
    _CRITIC_GRAD_CLIP: float = 100.0
    _EMA_DECAY: float = 0.98

    _IMAGINE_HORIZON: int = 13
    _GAMMA: float = 0.99
    _LAM: float = 0.95

    _CHECKPOINT_EVERY: int = 10_000

    def __init__(self, cfg: Any) -> None:
        """
        cfg: Hydra DictConfig or any object with attribute access.
        All phase-specific hyperparameters are read from cfg to support
        config-driven ablations — no hardcoded phase logic beyond the stubs above.
        """
        self.cfg = cfg
        self.device = self._resolve_device()
        self.step = 0

        wm_cfg = cfg.world_model
        self.world_model = TrikaWorldModel(
            obs_dim=wm_cfg.obs_dim,
            action_dim=wm_cfg.action_dim,
            n_levels=wm_cfg.levels,
            hidden_dim=wm_cfg.hidden_dim_apara,
            stoch_dim=wm_cfg.stoch_dim,
            stoch_classes=wm_cfg.stoch_classes,
            free_bits=wm_cfg.free_bits,
            kl_balance_dyn=wm_cfg.kl_balance_dyn,
            kl_balance_rep=wm_cfg.kl_balance_rep,
            decoder_z_only=getattr(wm_cfg, 'decoder_z_only', False),
        ).to(self.device)

        feature_dim = wm_cfg.hidden_dim_apara + wm_cfg.stoch_dim * wm_cfg.stoch_classes
        self._action_dim: int = int(wm_cfg.action_dim)  # cache as plain int to avoid Tensor|Module widening

        # Phase 2+: real EFE actor replaces zero-weight stub
        self.efe_actor = EFEActor(
            hidden_dim=wm_cfg.hidden_dim_apara,
            stoch_dim=wm_cfg.stoch_dim,
            n_cats=wm_cfg.stoch_classes,
            action_dim=wm_cfg.action_dim,
            n_layers=3,
        ).to(self.device)

        # Phase 2+: CRSPP preference model (SR-AIF creative value)
        self.crspp = CRSPPModel(
            hidden_dim=wm_cfg.hidden_dim_apara,
            gamma=self._GAMMA,
        ).to(self.device)

        self.critic = CriticHead(feature_dim).to(self.device)
        self.slow_critic = CriticHead(feature_dim).to(self.device)
        # Slow critic starts identical to fast critic
        self.slow_critic.load_state_dict(self.critic.state_dict())
        for p in self.slow_critic.parameters():
            p.requires_grad_(False)

        self.citta_store = CittaStore(
            hidden_dim=wm_cfg.hidden_dim_apara,
            n_levels=wm_cfg.levels,
        ).to(self.device)

        rew_cfg = cfg.reward
        self.camatk = CamatkaraReward(
            alpha_1=rew_cfg.alpha_1,
            alpha_2=rew_cfg.alpha_2,
            alpha_3=rew_cfg.alpha_3,
        )

        self.replay = ReplayBuffer(capacity=cfg.training.replay_capacity)

        loop_cfg = LoopConfig(
            sphuratta_vfe_percentile=cfg.sphuratta.percentile,
            sphuratta_min_gap=cfg.sphuratta.min_gap,
            camatk_alpha_vfe=rew_cfg.alpha_1,
            camatk_alpha_hopfield=rew_cfg.alpha_2,
            camatk_alpha_empowerment=rew_cfg.alpha_3,
            imagine_horizon=self._IMAGINE_HORIZON,
            gamma=self._GAMMA,
            lambda_=self._LAM,
            llm_enabled=cfg.llm.enabled,
            efe_enabled=True,   # Phase 2: EFEActor active
        )
        self.loop = PancakrtyaLoop(
            world_model=self.world_model,
            citta_store=self.citta_store,
            camatk=self.camatk,
            cfg=loop_cfg,
        )

        # Corpus environment selection (priority order):
        #   1. CachedCorpusEnv — if CORPUS_CACHE_DIR env var set and meta.json exists.
        #      ~100K step/sec; pre-embed once with: python -m pwm.data.embed_cache
        #   2. PhaseOneEnv    — live sentence-transformer embedding from .txt files.
        #      ~1 step/sec (sentence-transformer bottleneck per train step).
        #   3. TextEnv stub   — random observations, corpus not found.
        corpus_dir = Path(cfg.corpus.data_dir)
        cache_dir_env = os.environ.get("CORPUS_CACHE_DIR", "")
        _cache_dir = Path(cache_dir_env) if cache_dir_env else Path("data/embed_cache")
        _cache_ready = (_cache_dir / "meta.json").exists() and (_cache_dir / "embeddings.npy").exists()

        self._domain_selective: bool = bool(getattr(cfg, "domain_selective", False))
        if _cache_ready:
            if self._domain_selective:
                from pwm.data.embed_cache import DomainSelectiveCachedCorpusEnv  # type: ignore[import]
                log.info(
                    "DomainSelectiveCachedCorpusEnv (v5): action→domain coupling from %s", _cache_dir
                )
                self.env: Any = DomainSelectiveCachedCorpusEnv(
                    cache_dir=_cache_dir,
                    batch_size=cfg.training.batch_size,
                    seq_len=cfg.training.seq_len,
                    obs_dim=wm_cfg.obs_dim,
                    action_dim=wm_cfg.action_dim,
                    device=self.device,
                )
            else:
                from pwm.data.embed_cache import CachedCorpusEnv  # type: ignore[import]
                log.info("CachedCorpusEnv: loading pre-embedded corpus from %s", _cache_dir)
                self.env: Any = CachedCorpusEnv(
                    cache_dir=_cache_dir,
                    batch_size=cfg.training.batch_size,
                    seq_len=cfg.training.seq_len,
                    obs_dim=wm_cfg.obs_dim,
                    action_dim=wm_cfg.action_dim,
                    device=self.device,
                )
        elif corpus_dir.exists() and sum(1 for _ in corpus_dir.rglob('*.txt')) > 0:
            _txt_count = sum(1 for _ in corpus_dir.rglob('*.txt'))
            log.info('PhaseOneEnv: using real corpus at %s (%d files)', corpus_dir, _txt_count)
            log.info('TIP: run "python -m pwm.data.embed_cache --corpus-dir %s" for 100× speedup', corpus_dir)
            self.env = PhaseOneEnv(
                corpus_dir=corpus_dir,
                batch_size=cfg.training.batch_size,
                seq_len=cfg.training.seq_len,
                obs_dim=wm_cfg.obs_dim,
                action_dim=wm_cfg.action_dim,
                device=self.device,
                num_workers=0,
            )
        else:
            log.warning(
                'Corpus not found at %s — falling back to TextEnv stub. '
                'Set CORPUS_ROOT to real data or CORPUS_CACHE_DIR to embed cache.',
                corpus_dir,
            )
            self.env = TextEnv(
                batch_size=cfg.training.batch_size,
                obs_dim=wm_cfg.obs_dim,
                action_dim=wm_cfg.action_dim,
                seq_len=cfg.training.seq_len,
                device=self.device,
            )

        self.opt_wm = torch.optim.Adam(
            self.world_model.parameters(),
            lr=self._WM_LR,
            weight_decay=self._WM_WD,
            eps=1e-8,
        )
        # Phase 2+: joint actor + CRSPP optimisation
        self.opt_actor = torch.optim.Adam(
            list(self.efe_actor.parameters()) + list(self.crspp.parameters()),
            lr=self._ACTOR_LR,
            eps=1e-8,
        )
        self.opt_critic = torch.optim.Adam(
            self.critic.parameters(),
            lr=self._CRITIC_LR,
            eps=1e-8,
        )

        # bfloat16 mixed precision — native on GB10 Blackwell, no loss scaling needed
        mp_str: str = getattr(cfg.training, "mixed_precision", "bfloat16")
        self._mp_dtype = torch.bfloat16 if mp_str == "bfloat16" else torch.float32
        self._use_amp = mp_str in ("bfloat16", "float16")

        self._checkpoint_dir = Path("checkpoints")
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._setup_loggers()
        log.info("PWMTrainer initialised — %d parameters total", self._param_count())

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _param_count(self) -> int:
        models = [self.world_model, self.efe_actor, self.critic]
        return sum(p.numel() for m in models for p in m.parameters())

    def _setup_loggers(self) -> None:
        """Initialise wandb and mlflow only when env vars are set."""
        if _WANDB and os.environ.get("WANDB_PROJECT"):
            try:
                wandb.init(  # type: ignore[union-attr]
                    project=os.environ["WANDB_PROJECT"],
                    config=dict(self.cfg) if hasattr(self.cfg, "__iter__") else {},
                    resume="allow",
                )
                self._use_wandb = True
            except Exception:
                self._use_wandb = False
        else:
            self._use_wandb = False

        if _MLFLOW and os.environ.get("MLFLOW_TRACKING_URI"):
            try:
                mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])  # type: ignore[union-attr]
                mlflow.start_run()  # type: ignore[union-attr]
                self._use_mlflow = True
            except Exception:
                self._use_mlflow = False
        else:
            self._use_mlflow = False

    def _log_metrics(self, metrics: dict[str, float]) -> None:
        if self._use_wandb:
            wandb.log(metrics, step=self.step)  # type: ignore[union-attr]
        if self._use_mlflow:
            mlflow.log_metrics(metrics, step=self.step)  # type: ignore[union-attr]

    # ── Phase A: World Model ──────────────────────────────────────────────────

    # Imagination diversity loss weight — trains WM to produce different h_t
    # for different actions even in a passive (non-interactive) environment.
    # Without this, the corpus env's action-independence drives W_a → 0, making
    # prior entropy constant and the EFE advantage unmeasurable.
    _DIV_LOSS_WEIGHT: float = 0.05

    def _phase_a(self, batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> dict[str, float]:
        """
        Phase A: Variational Free Energy minimisation + imagination diversity.

        The world model observes a (B, T, D) sequence from replay and minimises
        VFE = KL(q||p) + reconstruction_loss across all active Trika levels.
        grad_clip=1000 (DreamerV3 §A.1) is large because the KL divergence can spike
        when the prior collapses; clipping prevents catastrophic gradient steps.

        Imagination Diversity Loss (IDL):
        Adds a contrastive term that penalises cosine similarity between h_t
        computed from two DIFFERENT random one-hot actions applied to the same
        initial state. This trains W_a (the action columns of input_proj) to be
        non-trivially non-zero, giving the EFE actor a signal it can exploit.

        Without IDL: VFE loss drives W_a → 0 (action is uninformative in corpus env).
        With IDL:    W_a learns to route action identity into distinct h_t directions.
        Weight 0.05 keeps IDL below VFE (~0.6) but large enough to maintain W_a signal.
        """
        obs_seq, action_seq, reward_seq, done_seq = batch
        B = obs_seq.shape[0]
        init_states = self.world_model.init_state(B, self.device)

        with torch.autocast(device_type=self.device.type, dtype=self._mp_dtype, enabled=self._use_amp):
            loss_dict = self.world_model.world_model_loss(
                obs_seq, action_seq, reward_seq, done_seq.float(), init_states
            )
            vfe_loss: Tensor = loss_dict["total"]  # type: ignore[assignment]

            # ── Imagination Diversity Loss ────────────────────────────────────
            # Sample two distinct random one-hot actions for each batch element.
            _adim = self._action_dim
            idx1 = torch.randint(0, _adim, (B,), device=self.device)
            idx2 = (idx1 + torch.randint(1, _adim, (B,), device=self.device)) % _adim  # guaranteed ≠ idx1
            a1 = nn.functional.one_hot(idx1, num_classes=_adim).float()  # (B, adim)
            a2 = nn.functional.one_hot(idx2, num_classes=_adim).float()  # (B, adim)

            # Single imagination step: different actions → should yield different h_t
            fresh_states = self.world_model.init_state(B, self.device)
            states1, _ = self.world_model.imagine_step(a1, fresh_states, step=0)
            states2, _ = self.world_model.imagine_step(a2, fresh_states, step=0)
            h1 = states1[0][0]  # (B, hidden_dim) — Aparā level h_t
            h2 = states2[0][0]  # (B, hidden_dim)

            # Diversity objective: minimise cosine similarity (maximise dissimilarity)
            cos_sim = nn.functional.cosine_similarity(h1, h2, dim=-1).mean()
            div_loss = cos_sim  # minimise → h1, h2 become more orthogonal

            wm_loss: Tensor = vfe_loss + self._DIV_LOSS_WEIGHT * div_loss

        self.opt_wm.zero_grad(set_to_none=True)
        wm_loss.backward()
        nn.utils.clip_grad_norm_(self.world_model.parameters(), self._WM_GRAD_CLIP)
        self.opt_wm.step()

        # Cache real VFE so Phase B/C imagination can use ΔF as camatkāra signal.
        self._last_real_vfe: float = float(vfe_loss.item())

        return {
            "loss/wm_total": float(wm_loss.item()),
            "loss/vfe": float(vfe_loss.item()),
            "loss/div": float(div_loss.item()),
            "train/action_cos_sim": float(cos_sim.item()),
            **{
                f"loss/{k}": float(v.item() if isinstance(v, Tensor) else v)
                for k, v in loss_dict.items()
                if k != "total"
            },
        }

    # ── Phase B: Actor ────────────────────────────────────────────────────────

    def _phase_b(self, start_states: list[tuple[Tensor, Tensor]]) -> dict[str, float]:
        """
        Phase B: Actor training over imagined rollouts.

        The actor imagines H=13 steps from the current WM state using the prior,
        collects camatkāra rewards along the imagined trajectory, then computes
        λ-returns and maximises them. The entropy term (svātantrya) prevents the
        policy from collapsing.

        grad_clip=100 is intentionally smaller than Phase A because the actor
        gradient signal is much lower-variance than the KL gradient.
        """
        H = self._IMAGINE_HORIZON
        B = start_states[0][0].shape[0]

        imag_states = start_states
        imag_h_list: list[Tensor] = []
        imag_z_list: list[Tensor] = []
        imag_rewards: list[Tensor] = []
        imag_dones: list[Tensor] = []

        _action_dim = self._action_dim
        with torch.autocast(device_type=self.device.type, dtype=self._mp_dtype, enabled=self._use_amp):
            for t in range(H):
                h_t, z_t = imag_states[0]
                # Phase 2+: EFEActor selects action from (h, z); embed as one-hot
                act_idx = self.efe_actor.select_action(h_t, z_t)          # (B,)
                action = torch.nn.functional.one_hot(act_idx, num_classes=_action_dim).float()
                imag_states, logits_prior_list = self.world_model.imagine_step(
                    action, imag_states, step=t
                )

                h_t, z_t = imag_states[0]
                imag_h_list.append(h_t)
                imag_z_list.append(z_t)

                # Camatkāra reward: prior-entropy VFE proxy (action-dependent surprise).
                #
                # Using negative prior entropy as `curr_vfe` so that:
                #   ΔF = max(H_prev_neg - H_curr_neg, 0) = max(ΔH_entropy_increase, 0)
                # i.e., reward fires when the WM prior GAINS entropy → actor navigated
                # the WM into a more uncertain/novel latent region (epistemic value).
                # This is fully action-dependent: different actions → different h_t
                # → different prior distributions → different entropy.
                # The EFE actor's epistemic objective exactly aligns with maximising this.
                #
                # IMPORTANT: z_t ~ Cat(32×32) means 32 INDEPENDENT Cat(32) distributions.
                # Use per-dimension log_softmax (dim=-1 over the 32 classes), NOT reshape
                # to 1024 which would model a single Cat(1024) — statistically wrong.
                lp_prior = logits_prior_list[0]  # (B, stoch_dim, stoch_classes)
                # Use Categorical.entropy() per-dimension — NaN-safe (handles p=0 via torch.where)
                # p*log_p manual form produces NaN in bfloat16 when policy peaks.
                prior_entropy_per_dim = Categorical(logits=lp_prior).entropy()   # (B, D)
                total_entropy_b = prior_entropy_per_dim.sum(-1).mean()     # scalar
                prior_neg_entropy = float(-total_entropy_b.item())
                # range: [−32·log32, 0] nats; more negative = higher entropy = more novel
                camatk_tensor, _ = self.camatk.compute(
                    curr_vfe=prior_neg_entropy,
                    hopfield_entropy_delta=self.citta_store.hopfield_entropy(level=0),
                    empowerment=0.0,
                )
                imag_rewards.append(camatk_tensor.to(self.device).expand(B))
                imag_dones.append(torch.zeros(B, dtype=torch.bool, device=self.device))

            # Stack trajectory: (B, H)
            rewards_t = torch.stack(imag_rewards, dim=1)
            dones_t = torch.stack(imag_dones, dim=1)
            feats_t = torch.stack(
                [torch.cat([h, z.flatten(-2)], dim=-1) for h, z in zip(imag_h_list, imag_z_list)],
                dim=1,
            )  # (B, H, feature_dim)

            # Values from slow critic (target network) — detached
            values_t = self.slow_critic.value(feats_t.detach())  # (B, H)
            returns_t = _compute_lambda_returns(rewards_t, values_t, dones_t, self._GAMMA, self._LAM)

            # Phase 2+: EFE actor loss — pg_loss + EFE minimisation + entropy bonus
            # imag_h_list / imag_z_list hold (B, hidden) and (B, D, K) tensors per step
            h_flat = torch.stack(imag_h_list, dim=1).reshape(B * H, -1)        # (B*H, hidden)
            z_for_actor = torch.stack(imag_z_list, dim=1)                       # (B, H, D, K)
            z_for_actor = z_for_actor.reshape(B * H, *z_for_actor.shape[2:])   # (B*H, D, K)
            advantage = returns_t.reshape(B * H)
            adv_std = advantage.std()
            if adv_std > 1e-6:
                advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)
            efe_losses = self.efe_actor.actor_loss(h_flat, z_for_actor, advantage)
            actor_loss = efe_losses["actor_total"]
            actor_entropy = efe_losses["entropy"]

        if not torch.isfinite(actor_loss):
            log.warning("Actor loss is NaN/Inf at step %d — skipping update", self.step)
            return {
                "loss/actor": float("nan"),
                "loss/actor_efe": float("nan"),
                "train/actor_entropy": float("nan"),
                "train/returns_mean": float(returns_t.mean().item()),
            }

        self.opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.efe_actor.parameters()) + list(self.crspp.parameters()),
            self._ACTOR_GRAD_CLIP,
        )
        self.opt_actor.step()

        return {
            "loss/actor": float(actor_loss.item()),
            "loss/actor_efe": float(efe_losses["efe_loss"].item()),
            "train/actor_entropy": float(actor_entropy.item()),
            "train/returns_mean": float(returns_t.mean().item()),
        }

    # ── Phase C: Critic ───────────────────────────────────────────────────────

    def _phase_c(self, start_states: list[tuple[Tensor, Tensor]]) -> dict[str, float]:
        """
        Phase C: Critic training and EMA slow-critic update.

        Twohot CE loss in symlog space prevents catastrophic over/under-estimation
        when the camatkāra reward distribution is heavy-tailed or bimodal.
        EMA slow critic (decay=0.98) provides a stable regression target.
        """
        H = self._IMAGINE_HORIZON
        B = start_states[0][0].shape[0]

        imag_states = start_states
        imag_feats: list[Tensor] = []
        imag_rewards: list[Tensor] = []
        imag_dones: list[Tensor] = []

        # Reset GRU hidden state before Phase C imagination to avoid dtype
        # mismatch: Phase B runs under bfloat16 autocast, Phase C does not.
        for level in self.world_model._level_list:
            level.sequence_model.reset_state()  # type: ignore[union-attr]

        with torch.no_grad():
            for t in range(H):
                h_t_c, z_t_c = imag_states[0]
                # Phase 2+: EFEActor.select_action(h, z) → discrete action index
                action = self.efe_actor.select_action(h_t_c, z_t_c)   # (B,)
                # Embed discrete action into continuous action_dim for WM step
                _adim = self._action_dim
                action_emb = torch.nn.functional.one_hot(action, num_classes=_adim).float()
                imag_states, logits_prior_list_c = self.world_model.imagine_step(
                    action_emb, imag_states, step=t
                )
                h_t, z_t = imag_states[0]
                imag_feats.append(torch.cat([h_t, z_t.flatten(-2)], dim=-1))

                # Same prior-entropy proxy as Phase B — NaN-safe Categorical.entropy().
                lp_prior_c = logits_prior_list_c[0]  # (B, D, K)
                prior_neg_entropy_c = float(
                    -Categorical(logits=lp_prior_c).entropy().sum(-1).mean().item()
                )
                camatk_tensor, _ = self.camatk.compute(
                    curr_vfe=prior_neg_entropy_c,
                    hopfield_entropy_delta=self.citta_store.hopfield_entropy(level=0),
                    empowerment=0.0,
                )
                imag_rewards.append(camatk_tensor.to(self.device).expand(B))
                imag_dones.append(torch.zeros(B, dtype=torch.bool, device=self.device))

        feats_t = torch.stack(imag_feats, dim=1)          # (B, H, D)
        rewards_t = torch.stack(imag_rewards, dim=1)       # (B, H)
        dones_t = torch.stack(imag_dones, dim=1)

        with torch.no_grad():
            values_t = self.slow_critic.value(feats_t)
            returns_t = _compute_lambda_returns(rewards_t, values_t, dones_t, self._GAMMA, self._LAM)

        with torch.autocast(device_type=self.device.type, dtype=self._mp_dtype, enabled=self._use_amp):
            critic_logits = self.critic(feats_t.reshape(B * H, -1))            # (B*H, N_BINS)
            targets_flat = returns_t.reshape(B * H)
            target_bins = twohot_encode(targets_flat, self.critic.bins)         # (B*H, N_BINS)
            critic_loss = twohot_loss(critic_logits, target_bins)

        self.opt_critic.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self._CRITIC_GRAD_CLIP)
        self.opt_critic.step()

        # EMA slow critic update
        _ema_update(self.slow_critic, self.critic, self._EMA_DECAY)

        return {
            "loss/critic": float(critic_loss.item()),
            "train/critic_value_mean": float(values_t.mean().item()),
        }

    # ── Public API ────────────────────────────────────────────────────────────

    def train_step(self, batch: tuple[Tensor, Tensor, Tensor, Tensor]) -> dict[str, float]:
        """
        Execute one full three-phase DreamerV3 training step.

        Args:
            batch: (obs_seq, action_seq, reward_seq, done_seq) each (B, T, D) or (B, T)
        Returns:
            metrics dict with all loss scalars for logging
        """
        self.world_model.train()
        self.efe_actor.train()
        self.critic.train()

        metrics: dict[str, float] = {}

        # Phase A — World Model
        metrics.update(self._phase_a(batch))

        # Initialise imagination start from a fresh WM state (not the replay batch)
        # to keep actor/critic imagination independent of the WM gradient graph.
        B = batch[0].shape[0]
        with torch.no_grad():
            start_states = self.world_model.init_state(B, self.device)

        # Phase B — Actor
        metrics.update(self._phase_b(start_states))

        # Phase C — Critic
        metrics.update(self._phase_c(start_states))

        metrics["train/step"] = float(self.step)
        return metrics

    def train(self, n_steps: int | None = None) -> dict[str, float]:
        """
        Main training loop. Runs for `n_steps` or `cfg.training.max_steps` if None.

        Collection phase fills the replay buffer with real env transitions before
        training starts (warm-up = min_buffer_steps). This prevents the WM from
        training on an empty buffer and getting degenerate KL gradients.
        """
        max_steps = n_steps if n_steps is not None else self.cfg.training.max_steps
        min_buf = self.cfg.training.min_buffer_steps
        log_interval = self.cfg.logging.log_interval
        validation_interval = self.cfg.training.eval_interval

        log.info("Filling replay buffer (min_buffer_steps=%d)...", min_buf)
        obs = self.env.reset()
        loop_state = self.loop.init(self.env.batch_size, self.device)

        # action_dim for replay storage. Store random one-hot actions so that the
        # WM's GRU action-conditioning weights receive gradient signal for all action
        # values, not just zero. This is critical: if the replay buffer contains only
        # zero actions, the GRU learns to ignore action inputs entirely, making the
        # prior entropy constant regardless of actor decisions (zero reward signal).
        _action_dim = self.cfg.world_model.action_dim
        while len(self.replay) < min_buf:
            B_env = obs.shape[0]
            rand_actions_np = np.eye(_action_dim, dtype=np.float32)[
                np.random.randint(_action_dim, size=B_env)
            ]  # (B, action_dim) random one-hot — consistent with train-loop convention
            if self._domain_selective:
                step_action = torch.tensor(rand_actions_np, device=self.device)
            else:
                step_action = loop_state.action
            _, reward, done = self.env.step(step_action)
            next_obs = obs  # stub: next obs same as obs for warm-up
            # Store each batch item as a separate transition so that
            # _collate_sequences assembles (B, T, obs_dim) correctly.
            for b in range(B_env):
                self.replay.add(Transition(
                    obs=obs[b].cpu().numpy(),
                    action=rand_actions_np[b],
                    reward=float(reward[b].item()),
                    done=bool(done[b].item()),
                    next_obs=next_obs[b].cpu().numpy(),
                    vfe=0.0,
                ))
            obs = next_obs

        log.info("Training for %d steps...", max_steps)
        t0 = time.perf_counter()
        done = torch.zeros(self.env.batch_size, dtype=torch.bool, device=self.device)
        metrics: dict[str, float] = {}

        while self.step < max_steps:
            # Sample a sequence batch from replay
            seqs = self.replay.sample_sequence(
                seq_len=self.cfg.training.seq_len,
                batch_size=self.cfg.training.batch_size,
            )
            if not seqs:
                # Buffer ran dry — collect more transitions
                _, reward, done = self.env.step(loop_state.action)
                continue

            batch = self._collate_sequences(seqs)
            metrics = self.train_step(batch)

            # Collect one real transition per train step to keep buffer fresh.
            # Store each batch item separately so _collate_sequences sees (obs_dim,) obs.
            #
            # Phase 2 v5 (domain_selective): generate the random action BEFORE stepping
            # so the env uses it to pick the corpus domain.  This creates a genuine
            # causal chain: rand_action_t → obs_{t+1} ∈ domain(rand_action_t).
            # The WM then trains on (obs_t, rand_action_t, obs_{t+1}) sequences where
            # obs_{t+1} is NOT independent of action_t — breaking Layer-3 passivity.
            #
            # Passive mode: use _zero_action (PancakrtyaLoop-compatible shape).
            B_env = obs.shape[0]
            _vfe = metrics.get("loss/wm_total", 0.0)
            rand_actions_np = np.eye(_action_dim, dtype=np.float32)[
                np.random.randint(_action_dim, size=B_env)
            ]  # (B, action_dim) random one-hot for each batch item
            if self._domain_selective:
                # Pass one-hot actions so DomainSelectiveCachedCorpusEnv can decode them
                step_action = torch.tensor(rand_actions_np, device=self.device)
            else:
                step_action = torch.zeros(B_env, 1, device=self.device)
            obs_new, reward, done = self.env.step(step_action)
            for b in range(B_env):
                self.replay.add(Transition(
                    obs=obs[b].cpu().numpy(),
                    action=rand_actions_np[b],
                    reward=float(reward[b].item()),
                    done=bool(done[b].item()),
                    next_obs=obs_new[b].cpu().numpy(),
                    vfe=_vfe,
                ))
            obs = obs_new

            self.step += 1

            if self.step % log_interval == 0:
                elapsed = time.perf_counter() - t0
                sps = self.step / elapsed
                log.info(
                    "step=%d  wm=%.4f  vfe=%.4f  div=%.4f  cos_sim=%.3f  actor=%.4f  critic=%.4f  sps=%.1f",
                    self.step,
                    metrics.get("loss/wm_total", 0.0),
                    metrics.get("loss/vfe", metrics.get("loss/wm_total", 0.0)),
                    metrics.get("loss/div", 0.0),
                    metrics.get("train/action_cos_sim", 0.0),
                    metrics.get("loss/actor", 0.0),
                    metrics.get("loss/critic", 0.0),
                    sps,
                )
                self._log_metrics({**metrics, "train/steps_per_sec": sps})

            if self.step % self._CHECKPOINT_EVERY == 0:
                ckpt_path = self._checkpoint_dir / f"step_{self.step:07d}.pt"
                self.save_checkpoint(ckpt_path)

            if self.step % validation_interval == 0:
                self._run_validation()

        log.info("Training complete at step %d.", self.step)
        self.save_checkpoint(self._checkpoint_dir / "final.pt")

        if _MLFLOW and self._use_mlflow:
            mlflow.end_run()  # type: ignore[union-attr]

        return metrics

    def _collate_sequences(
        self, seqs: list[list[Transition]]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Convert a list of Transition sequences into (B, T, D) training tensors.
        Rewards are zeroed here — camatkāra reward is applied during actor/critic phases.
        """
        obs_list, act_list, rew_list, done_list = [], [], [], []
        for seq in seqs:
            obs_list.append(np.stack([t.obs for t in seq]))     # (T, obs_dim)
            act_list.append(np.stack([t.action for t in seq]))  # (T, action_dim)
            rew_list.append(np.array([t.reward for t in seq]))  # (T,)
            done_list.append(np.array([t.done for t in seq]))   # (T,)

        obs_t = torch.tensor(np.stack(obs_list), dtype=torch.float32, device=self.device)
        act_t = torch.tensor(np.stack(act_list), dtype=torch.float32, device=self.device)
        rew_t = torch.tensor(np.stack(rew_list), dtype=torch.float32, device=self.device)
        done_t = torch.tensor(np.stack(done_list), dtype=torch.bool, device=self.device)
        return obs_t, act_t, rew_t, done_t

    def _run_validation(self) -> None:
        """Validation pass — logs VFE on a held-out batch. Extended in Phase 1+."""
        self.world_model.eval()
        with torch.no_grad():
            obs_seq, action_seq, reward_seq, done_seq = self.env.sample_batch()
            init_states = self.world_model.init_state(obs_seq.shape[0], self.device)
            loss_dict = self.world_model.world_model_loss(
                obs_seq, action_seq, reward_seq, done_seq.float(), init_states
            )
            vfe = float(loss_dict["total"].item())  # type: ignore[union-attr]
        log.info("validation step=%d  VFE=%.4f", self.step, vfe)
        self._log_metrics({"validation/vfe": vfe})
        self.world_model.train()

    def save_checkpoint(self, path: Path | str) -> None:
        """Save full trainer state for exact resumption."""
        path = Path(path)
        torch.save(
            {
                "step": self.step,
                "world_model": self.world_model.state_dict(),
                "efe_actor": self.efe_actor.state_dict(),
                "critic": self.critic.state_dict(),
                "slow_critic": self.slow_critic.state_dict(),
                "citta_store": self.citta_store.state_dict(),
                "opt_wm": self.opt_wm.state_dict(),
                "opt_actor": self.opt_actor.state_dict(),
                "opt_critic": self.opt_critic.state_dict(),
            },
            path,
        )
        log.info("Checkpoint saved: %s", path)

    def load_checkpoint(self, path: Path | str) -> None:
        """Resume from a saved checkpoint, including optimizer states."""
        path = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.step = ckpt["step"]
        self.world_model.load_state_dict(ckpt["world_model"])
        self.efe_actor.load_state_dict(ckpt["efe_actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.slow_critic.load_state_dict(ckpt["slow_critic"])
        self.citta_store.load_state_dict(ckpt["citta_store"])
        self.opt_wm.load_state_dict(ckpt["opt_wm"])
        self.opt_actor.load_state_dict(ckpt["opt_actor"])
        self.opt_critic.load_state_dict(ckpt["opt_critic"])
        log.info("Checkpoint loaded from %s (step=%d)", path, self.step)


# ── Entry point ───────────────────────────────────────────────────────────────

if _HYDRA:
    @hydra.main(  # type: ignore[misc]
        config_path="../../configs",
        config_name="default",
        version_base=None,
    )
    def main(cfg: DictConfig) -> None:  # type: ignore[misc,no-redef]
        """
        Hydra entry point. All config overrides go through `--config-name` or `+key=val`.
        See `configs/` for phase-specific presets.
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        log.info("Config:\n%s", OmegaConf.to_yaml(cfg))  # type: ignore[possibly-unbound]

        torch.manual_seed(cfg.training.seed)
        np.random.seed(cfg.training.seed)

        trainer = PWMTrainer(cfg)

        resume_ckpt = os.environ.get("PWM_RESUME_CHECKPOINT")
        if resume_ckpt and Path(resume_ckpt).exists():
            trainer.load_checkpoint(resume_ckpt)

        # Cross-phase warm-start: load WM weights only from a Phase 1 checkpoint.
        # Leaves efe_actor/critic at random init (Phase 2 trains these from scratch
        # on top of the pre-trained world model substrate).
        wm_ckpt = os.environ.get("PWM_RESUME_WM_ONLY")
        if wm_ckpt and Path(wm_ckpt).exists() and not resume_ckpt:
            ckpt = torch.load(wm_ckpt, map_location=trainer.device, weights_only=False)
            # strict=False ignores missing/unexpected keys but still raises on shape
            # mismatches. Filter by shape first so the decoder (1536→1024 in v7) is
            # skipped and freshly initialised while encoder/prior/GRU are loaded.
            ckpt_wm_sd = ckpt["world_model"]
            model_sd = trainer.world_model.state_dict()
            filtered_sd = {k: v for k, v in ckpt_wm_sd.items()
                           if k in model_sd and v.shape == model_sd[k].shape}
            skipped = [k for k in ckpt_wm_sd if k not in filtered_sd]
            if skipped:
                log.info("WM warm-start: skipping %d shape-mismatched keys (fresh init): %s",
                         len(skipped), skipped[:5])
            missing, unexpected = trainer.world_model.load_state_dict(filtered_sd, strict=False)
            if missing:
                log.info("WM warm-start: %d keys not in checkpoint (fresh init): %s",
                         len(missing), missing[:5])
            if unexpected:
                log.info("WM warm-start: %d unexpected keys (ignored): %s",
                         len(unexpected), unexpected[:5])
            log.info("Loaded WM-only from checkpoint: %s (step=%d)", wm_ckpt, ckpt.get("step", -1))

        trainer.train()
else:
    def main() -> None:  # type: ignore[misc,no-redef]
        raise RuntimeError(
            "Hydra is not installed. Install it with: pip install hydra-core"
        )


if __name__ == "__main__":
    main()
