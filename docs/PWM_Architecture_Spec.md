# The Pratyabhijñā World Model (PWM): Architecture Specification
## Technical Reference for DGX Spark 128GB Implementation
### v1.0 — May 2026

---

> **Scope.** This document is the implementation-grade technical specification for the PWM system. An ML engineer with a PyTorch background should be able to reproduce the full system from this document plus the cited open-source libraries. It is a companion to `PWM_Master_Research.md` (which provides the research rationale) and `PWM_PRD_Plan.md` (which provides the product requirements and phased plan).

---

## 1. System Overview

The PWM is a two-tier creative AI system:

```
┌─────────────────────────────────────────────────────────────────────┐
│  TIER 3: MULTI-AGENT ORCHESTRATION (Pañcakṛtya Pipeline)            │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  smolagents: cit→ānanda→icchā→apohana→jñāna→kriyā→vimarśa   │   │
│  │  Avacchedaka store (Pratyākṣa PCEH): typed inter-agent msgs  │   │
│  │  Sākṣī-keeper: ≤500-token witness invariant (cross-cutting)  │   │
│  │  Memory-agent + Sleep-agent: CittaStore + NREM/REM            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↕ LiteLLM (TRT-LLM / vLLM local)       │
│  TIER 2: LLM ĀGAMA LAYER (Conscious / Knowledge)                    │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Nemotron 3 Super 120B A12B MoE (12B active, ~44GB FP4)      │   │
│  │  [TensorRT-LLM on DGX Spark GB10 Blackwell FP4 tensor cores] │   │
│  │  + Nemotron-Super-49B Dense (~28GB FP8, vLLM) fast sub-agents│   │
│  │  LoRA vimarśa bridge ← → WM latent projection                │   │
│  │  Goal specification: user_intent → preference C              │   │
│  │  Sphurattā narration: C_t=1 → skill library entry            │   │
│  │  DECKARD AWM proposals: long-horizon plan hypotheses         │   │
│  │  Human interface: natural language I/O                       │   │
│  │  Zero paid API calls (svātantrya principle)                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↕ vimarśa bridge                        │
│  TIER 1: WORLD MODEL SUBSTRATE (Subconscious / Prakāśa)             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Para level (S4, stride 16) — global, slow                   │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  Parāparā level (GRU, stride 4) — coupling, mid     │    │   │
│  │  │  ┌──────────────────────────────────────────────┐   │    │   │
│  │  │  │  Aparā level (GRU, stride 1) — fast, embodied │   │    │   │
│  │  │  │  RSSM: h_t, z_t (32×32 cat)                  │   │    │   │
│  │  │  │  EFE actor + distributional critic             │   │    │   │
│  │  │  └──────────────────────────────────────────────┘   │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │  Hopfield Citta-store (episodic + semantic, per level)       │   │
│  │  Sleep scheduler (NREM + REM phases)                         │   │
│  │  Camatkāra detector + intrinsic reward                       │   │
│  │  Prioritized replay buffer                                   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              ↕ encoder/decoder                       │
│  PERCEPTION FRONT-END                                                │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  v1.0 (TEXT ONLY): BPE tokenizer + linear embedding           │   │
│  │  v2.0+: V-JEPA 2 (frozen, 1.2B ViT) — visual domain          │   │
│  │  Phase 5+: DIAMOND EDM decoder — high-fidelity output         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Tier 1: World Model Core

### 2.1 RSSM Specification (Per Level)

Each Trika level instantiates a full RSSM:

```python
class TrikaCoreLevel(nn.Module):
    """
    One level of the three-level Trika RSSM hierarchy.
    Level 0 (Aparā): stride=1, GRU backbone
    Level 1 (Parāparā): stride=4, GRU backbone
    Level 2 (Para): stride=16, S4 backbone (R2I recipe)
    """
    def __init__(
        self,
        level: int,                # 0, 1, 2
        obs_dim: int,              # observation/feature dimension
        stoch_dim: int = 32,       # categorical variable count
        stoch_classes: int = 32,   # classes per variable
        hidden_dim: int = 512,     # GRU/S4 hidden dim (512 for level 0, 1024 for 2)
        action_dim: int = 64,      # action embedding dim
        backbone: str = 'gru',     # 'gru' or 's4'
    ):
        # Encoder: q_φ(z_t | h_t, o_t) — the recognition density (pratyabhijñā)
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )
        
        # Recurrent backbone: h_t = f(h_{t-1}, z_{t-1}, a_{t-1}) — spanda dynamics
        if backbone == 'gru':
            self.sequence_model = nn.GRUCell(
                input_size=stoch_dim * stoch_classes + action_dim,
                hidden_size=hidden_dim,
            )
        elif backbone == 's4':
            from s4 import S4  # from state-spaces/s4 repo
            self.sequence_model = S4(
                d_model=hidden_dim,
                d_state=64,
                bidirectional=False,  # causal
                dropout=0.0,
            )
        
        # Prior: p_θ(z_t | h_t) — the prior latent (cit level)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, stoch_dim * stoch_classes),
        )
        
        # Decoder: p_θ(o_t | h_t, z_t) — generative model
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + stoch_dim * stoch_classes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, obs_dim),
        )
        
        # Reward head: p_θ(r_t | h_t, z_t)
        self.reward_head = SymlogTwohotHead(hidden_dim + stoch_dim * stoch_classes)
        
        # Continue head: p_θ(c_t | h_t, z_t)
        self.continue_head = nn.Linear(hidden_dim + stoch_dim * stoch_classes, 1)
        
        # Camatkāra / sphurattā components
        self.vfe_tracker = RollingWindowStats(window=100)
        self.sphuratta_threshold_percentile = 5  # fire on bottom 5% VFE
        
    def observe(self, obs, prev_h, prev_z, prev_a):
        """Recognition step: q_φ(z_t | h_t, o_t) — pratyabhijñā."""
        h_t = self.sequence_model(
            torch.cat([prev_z.flatten(-2), prev_a], dim=-1), prev_h
        )
        # Posterior (recognition density)
        logits_post = self.encoder(torch.cat([obs, h_t], dim=-1))
        z_t = straight_through_sample(logits_post.reshape(-1, self.stoch_dim, self.stoch_classes))
        # Prior (for KL computation)
        logits_prior = self.prior(h_t)
        return h_t, z_t, logits_post, logits_prior
    
    def imagine(self, prev_h, prev_z, prev_a):
        """Imagination step: ẑ_t ~ p_θ(z_t | h_t) — pure prior."""
        h_t = self.sequence_model(
            torch.cat([prev_z.flatten(-2), prev_a], dim=-1), prev_h
        )
        logits_prior = self.prior(h_t)
        z_t = straight_through_sample(logits_prior.reshape(-1, self.stoch_dim, self.stoch_classes))
        return h_t, z_t, logits_prior
    
    def compute_vfe(self, logits_post, logits_prior):
        """
        Variational Free Energy: F = KL[q_φ ‖ p_θ] (complexity) + reconstruction_loss (accuracy)
        Implementation follows DreamerV3's free-bits KL balancing.
        """
        kl = kl_categorical_free_bits(logits_post, logits_prior, free_bits=1.0)
        return kl
```

### 2.2 Hierarchical Integration

```python
class TrikaWorldModel(nn.Module):
    """
    Three-level Trika hierarchy.
    Levels are conditionally dependent: upper levels condition lower levels.
    """
    def __init__(self, obs_dim, action_dim):
        self.levels = nn.ModuleList([
            TrikaCoreLevel(level=0, obs_dim=obs_dim, hidden_dim=512, backbone='gru'),    # Aparā
            TrikaCoreLevel(level=1, obs_dim=512, hidden_dim=1024, backbone='gru'),       # Parāparā
            TrikaCoreLevel(level=2, obs_dim=1024, hidden_dim=1024, backbone='s4'),       # Para
        ])
        self.level_upsampler = nn.ModuleList([
            nn.Linear(1024, 512),  # Para → Parāparā conditioning
            nn.Linear(1024, 512),  # Parāparā → Aparā conditioning
        ])
        self.hopfield_stores = nn.ModuleList([
            CittaStore(latent_dim=512+32*32, n_episodic=1000, n_semantic=64),
            CittaStore(latent_dim=1024+32*32, n_episodic=500, n_semantic=32),
            CittaStore(latent_dim=1024+32*32, n_episodic=100, n_semantic=16),
        ])
    
    def forward(self, obs_sequence):
        """Process sequence at all levels with temporal downsampling."""
        outputs = {}
        # Level 0 (Aparā): process every token/frame
        for t, obs_t in enumerate(obs_sequence):
            # Condition on Para and Parāparā via top-down signals
            para_signal = self._get_para_signal(t)
            para_para_signal = self._get_para_para_signal(t)
            obs_conditioned = obs_t + para_signal + para_para_signal
            
            # Hopfield retrieval (retrieval-augmented imagination)
            c_t = self.hopfield_stores[0].retrieve(self.states[0]['h_t'], self.states[0]['z_t'])
            obs_augmented = torch.cat([obs_conditioned, c_t], dim=-1)
            
            h_t, z_t, lp, lq = self.levels[0].observe(obs_augmented, ...)
            
            # Camatkāra detection
            vfe_t = self.levels[0].compute_vfe(lp, lq)
            if self.hopfield_stores[0].is_sphuratta(h_t, z_t, vfe_t):
                self.emit_sphuratta_event(level=0, h_t=h_t, z_t=z_t, t=t)
            
            # Every stride-4 steps: update Parāparā
            if t % 4 == 0:
                self._update_para_para(t)
            # Every stride-16 steps: update Para
            if t % 16 == 0:
                self._update_para(t)
```

### 2.3 Training Objective

```python
def world_model_loss(self, batch):
    """
    DreamerV3-style VFE loss, isomorphic to Friston's variational free energy.
    L = L_pred (accuracy) + β_dyn * L_dyn + β_rep * L_rep (complexity)
    """
    obs, actions, rewards, dones = batch
    
    # Forward pass (observe phase)
    h_seq, z_seq, logits_post, logits_prior = self.observe_sequence(obs, actions)
    
    # Prediction losses (accuracy term: -E_Q[log p(o,r,c | h,z)])
    obs_pred = self.decoder(torch.cat([h_seq, z_seq], dim=-1))
    L_obs = F.mse_loss(symlog(obs_pred), symlog(obs))  # symlog MSE
    
    reward_pred = self.reward_head(torch.cat([h_seq, z_seq], dim=-1))
    L_reward = twohot_loss(reward_pred, symlog(rewards))
    
    continue_pred = self.continue_head(torch.cat([h_seq, z_seq], dim=-1))
    L_continue = F.binary_cross_entropy_with_logits(continue_pred, 1.0 - dones)
    
    L_pred = L_obs + L_reward + L_continue
    
    # KL losses (complexity term: D_KL[Q ‖ P])
    # KL balancing: separate gradients for dynamics and representation
    L_dyn = kl_categorical_free_bits(
        logits_post.detach(), logits_prior, free_bits=1.0
    )  # trains the prior p_θ
    L_rep = kl_categorical_free_bits(
        logits_post, logits_prior.detach(), free_bits=1.0
    )  # trains the encoder q_φ
    
    # Total loss
    L_total = L_pred + β_dyn * L_dyn + β_rep * L_rep
    # β_dyn = 0.5, β_rep = 0.1 (DreamerV3 defaults)
    
    # Log VFE for camatkāra detection
    vfe_per_step = (L_dyn + L_rep).detach()
    self.update_vfe_statistics(vfe_per_step)
    
    return L_total
```

---

## 3. Active Inference Actor (EFE Minimisation)

### 3.1 Expected Free Energy Formulation

Replacing DreamerV3's REINFORCE actor with EFE minimisation (following Tschantz et al. 2020; SR-AIF 2025):

```python
class EFEActor(nn.Module):
    """
    Active inference actor that minimises Expected Free Energy G(π).
    G(π) = ambiguity + risk - epistemic_value - parameter_novelty
    """
    def __init__(self, latent_dim, action_dim, horizon=15):
        self.policy_net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, action_dim * 2),  # mean + log_std
        )
        self.value_net = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.SiLU(),
            nn.Linear(512, 512), nn.SiLU(),
            nn.Linear(512, 255),  # twohot distributional
        )
        # CRSPP: Contrastive Recurrent State Prior Preference (SR-AIF)
        self.preference_model = CRSPPPreference(latent_dim)
        self.horizon = horizon
        
    def compute_efe(self, world_model, h_0, z_0, preference_C, n_samples=16):
        """
        Compute EFE for candidate policies by imagining rollouts.
        G(π) = Σ_τ [ambiguity(τ) + risk(τ) - epistemic(τ) - novelty(τ)]
        """
        all_efe = []
        
        for _ in range(n_samples):
            h_t, z_t = h_0.clone(), z_0.clone()
            efe_trajectory = 0.0
            
            for τ in range(self.horizon):
                # Sample action from policy
                latent = torch.cat([h_t, z_t.flatten(-2)], dim=-1)
                mu, log_std = self.policy_net(latent).chunk(2, dim=-1)
                a_t = mu + torch.randn_like(mu) * log_std.exp()
                
                # Imagine next step (prior only — no encoder)
                h_next, z_next, logits_prior = world_model.imagine(h_t, z_t, a_t)
                
                # ─── EFE terms ───
                
                # 1. Ambiguity: H[p(o | s)] — expected observation entropy under prior
                obs_pred = world_model.decoder(torch.cat([h_next, z_next], dim=-1))
                # Use decoder variance as ambiguity proxy
                ambiguity = world_model.decoder_variance(h_next, z_next).mean()
                
                # 2. Risk: D_KL[Q(o|π) ‖ P(o|C)] — deviation from preference
                pred_log_prob = world_model.reward_head.log_prob(obs_pred)
                pref_log_prob = self.preference_model(z_next, preference_C)
                risk = (pred_log_prob - pref_log_prob).mean()
                
                # 3. Epistemic value: D_KL[Q(z|o,π) ‖ Q(z|π)]
                # Approximate: information gain from imagined observation
                # Sample a pseudo-observation from the decoder
                o_pseudo = world_model.decoder.sample(h_next, z_next)
                logits_post_pseudo, _ = world_model.encode(o_pseudo, h_next)
                epistemic = kl_categorical(logits_post_pseudo, logits_prior).mean()
                
                # 4. Parameter novelty: information gain about model parameters
                # Approximate with the disagreement across ensemble heads
                novelty = world_model.ensemble_disagreement(h_next, z_next)
                
                # EFE (lower = better; agent minimises this)
                efe_τ = ambiguity + risk - epistemic - novelty
                
                # Apply discount
                efe_trajectory += (self.gamma ** τ) * efe_τ
                
                h_t, z_t = h_next, z_next
            
            all_efe.append(efe_trajectory)
        
        return torch.stack(all_efe).mean(0)
    
    def act(self, h_t, z_t, preference_C, world_model):
        """Sample action that minimises expected free energy."""
        latent = torch.cat([h_t, z_t.flatten(-2)], dim=-1)
        mu, log_std = self.policy_net(latent).chunk(2, dim=-1)
        
        # During imagination: use EFE gradient
        # During execution: sample from policy + entropy regularisation
        a = mu + torch.randn_like(mu) * log_std.exp()
        return a
    
    def actor_loss(self, world_model, replay_batch):
        """
        Train actor to minimise EFE, following SR-AIF actor-critic recipe.
        Uses lambda-returns on EFE rather than reward.
        """
        # Imagine from replay states
        with torch.no_grad():
            imagined_traj = world_model.imagine_rollout(
                replay_batch['h'], replay_batch['z'],
                actor=self, horizon=self.horizon
            )
        
        # Compute EFE targets
        efe_targets = self.compute_lambda_efe_targets(imagined_traj)
        
        # Actor loss: minimise EFE
        log_probs = self.policy_log_probs(imagined_traj['actions'], imagined_traj['latents'])
        entropy = self.policy_entropy(imagined_traj['latents'])
        
        actor_loss = -(log_probs * efe_targets.detach()).mean() - η_entropy * entropy.mean()
        
        # Critic loss: fit value net to EFE
        efe_pred = self.value_net(imagined_traj['latents'])
        critic_loss = twohot_loss(efe_pred, symlog(efe_targets))
        
        return actor_loss + critic_loss
```

### 3.2 CRSPP Preference Model

```python
class CRSPPPreference(nn.Module):
    """
    Contrastive Recurrent State Prior Preference (SR-AIF, ICRA 2025).
    Learns what the agent 'wants' from demonstrations and successful episodes.
    This is the computational C (preference distribution) in EFE.
    """
    def __init__(self, latent_dim):
        self.goal_encoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.state_encoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.SiLU(),
            nn.Linear(256, 128),
        )
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, z_state, goal_context):
        """
        Returns log p(o|C) — log probability of state under preference.
        High when state matches goal, low when it doesn't.
        """
        s = F.normalize(self.state_encoder(z_state), dim=-1)
        g = F.normalize(self.goal_encoder(goal_context), dim=-1)
        return (s * g).sum(-1) / self.temperature
    
    def contrastive_loss(self, positive_pairs, negative_pairs):
        """InfoNCE loss for preference learning."""
        pos_sim = self(positive_pairs['z'], positive_pairs['goal'])
        neg_sim = self(negative_pairs['z'], negative_pairs['goal'])
        return F.cross_entropy(
            torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1),
            torch.zeros(len(pos_sim), dtype=torch.long)
        )
```

---

## 4. Hopfield Citta-Store

### 4.1 Two-Mode Store per Level

```python
class CittaStore(nn.Module):
    """
    Hopfield Citta-store: episodic (smṛti) + semantic (ālayavijñāna).
    Two modes with different β values for different retrieval behaviours.
    """
    def __init__(self, latent_dim, n_episodic=1000, n_semantic=64):
        from hflayers import Hopfield, HopfieldLayer
        
        self.latent_dim = latent_dim
        self.n_episodic = n_episodic
        
        # Episodic: FIFO over recent experiences, high β (sharp recall)
        self.hopfield_episodic = Hopfield(
            input_size=latent_dim,
            hidden_size=latent_dim,
            output_size=latent_dim // 2,
            num_heads=8,
            scaling=4.0 / (latent_dim ** 0.5),  # 4× standard = high β
            update_steps_max=3,
            dropout=0.0,
        )
        
        # Semantic: learnable prototypes, low β (concept blending)
        self.hopfield_semantic = HopfieldLayer(
            input_size=latent_dim,
            hidden_size=latent_dim,
            output_size=latent_dim // 2,
            num_heads=4,
            scaling=0.25 / (latent_dim ** 0.5),  # 0.25× = low β
            stored_pattern_size=latent_dim,
            pattern_projection_as_static=False,
            normalize_stored_pattern=True,
            normalize_state_pattern=True,
        )
        
        # FIFO buffer
        self.episodic_buffer = []
        self.buffer_size = n_episodic
        
        # Sphurattā detector
        self.retrieval_entropy_history = deque(maxlen=200)
        self.sphuratta_threshold = None  # set from running percentile
        
    def write_episodic(self, h_t, z_t, reward, novelty):
        """Online episodic write at each step."""
        entry = torch.cat([h_t, z_t.flatten(-2)], dim=-1)
        self.episodic_buffer.append({
            'latent': entry.detach(),
            'reward': reward,
            'novelty': novelty,
            'timestamp': self.step_count,
        })
        if len(self.episodic_buffer) > self.buffer_size:
            self.episodic_buffer.pop(0)  # FIFO
    
    def retrieve(self, h_t, z_t):
        """Retrieve context for current state — retrieval-augmented imagination."""
        query = torch.cat([h_t, z_t.flatten(-2)], dim=-1).unsqueeze(0)
        
        # Episodic context
        if len(self.episodic_buffer) > 0:
            stored = torch.stack([e['latent'] for e in self.episodic_buffer]).unsqueeze(0)
            c_episodic = self.hopfield_episodic(stored, stored, query).squeeze(0)
        else:
            c_episodic = torch.zeros(self.latent_dim // 2, device=h_t.device)
        
        # Semantic context (always available after Phase 3+)
        c_semantic = self.hopfield_semantic(query).squeeze(0)
        
        # Track retrieval entropy for sphurattā detection
        with torch.no_grad():
            # Compute entropy of attention weights
            attn_weights = self._get_last_attention_weights()
            H_retrieval = -(attn_weights * attn_weights.clamp(min=1e-8).log()).sum(-1).mean()
            self.retrieval_entropy_history.append(H_retrieval.item())
            self._update_sphuratta_threshold()
        
        return torch.cat([c_episodic, c_semantic], dim=-1)
    
    def is_sphuratta(self, h_t, z_t, vfe_t):
        """
        Sphurattā event detection: recognition flash.
        Fires when BOTH VFE drops sharply AND Hopfield retrieval entropy drops.
        """
        if self.sphuratta_threshold is None:
            return False
        H_current = self.retrieval_entropy_history[-1] if self.retrieval_entropy_history else float('inf')
        vfe_drop = self.vfe_history[-1] - vfe_t  # positive = VFE decreased
        
        return (
            H_current < self.sphuratta_threshold  # Hopfield convergence
            and vfe_drop > self.vfe_drop_threshold  # Free energy drop
            and (self.last_sphuratta_step is None 
                 or self.step_count - self.last_sphuratta_step > self.min_sphuratta_gap)
        )
    
    def consolidate_sws(self, world_model):
        """
        NREM-analog: slow-wave sleep consolidation.
        - Replay prioritised episodes
        - Down-scale un-accessed patterns (synaptic homeostasis)
        - Distill well-modelled patterns into semantic store
        """
        # Identify patterns not accessed in recent steps
        recent_accessed = set(self.recently_accessed_indices)
        for i, entry in enumerate(self.episodic_buffer):
            if i not in recent_accessed:
                # Synaptic homeostasis: reduce weight
                entry['latent'] *= (1 - self.shsy_rate)
        
        # Distill converged patterns into semantic HopfieldLayer
        high_recognition = [
            e['latent'] for e in self.episodic_buffer
            if e['novelty'] < self.consolidation_threshold
        ]
        if len(high_recognition) > 8:
            self._update_semantic_prototypes(torch.stack(high_recognition))
    
    def consolidate_rem(self, dream_latents):
        """
        REM-analog: cluster dream latents and add as semantic prototypes.
        """
        # K-means clustering of dream latents
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=min(8, len(dream_latents)))
        km.fit(dream_latents.cpu().numpy())
        centroids = torch.tensor(km.cluster_centers_, dtype=dream_latents.dtype)
        self._update_semantic_prototypes(centroids.to(dream_latents.device))
```

---

## 5. Vimarśa Bridge (LLM Integration)

### 5.1 Architecture

```python
class VimarsaBridge(nn.Module):
    """
    Bidirectional bridge between world model latents and frozen LLM.
    - WM → LLM: project latent state into LLM context for narration
    - LLM → WM: project LLM hidden states into WM preference space
    
    Implemented as LoRA-scale cross-attention projections.
    """
    def __init__(self, wm_dim, llm_dim, bridge_dim=256):
        # WM latent → LLM token embedding projection
        self.wm_to_llm = nn.Sequential(
            nn.Linear(wm_dim, bridge_dim), nn.SiLU(),
            nn.Linear(bridge_dim, llm_dim),  # project to LLM token dim
        )
        
        # LLM hidden → WM preference projection
        self.llm_to_wm = nn.Sequential(
            nn.Linear(llm_dim, bridge_dim), nn.SiLU(),
            nn.Linear(bridge_dim, wm_dim),
        )
        
        # LoRA adapters (trained; LLM weights frozen)
        self.lora_q = nn.Linear(llm_dim, 16, bias=False)  # rank 16
        self.lora_v = nn.Linear(16, llm_dim, bias=False)
        
    def narrate_latent(self, h_t, z_t, llm, prompt_template):
        """
        At a sphurattā event: project latent state into LLM context,
        receive a natural-language narration of the recognition event.
        Called ONLY at sphurattā events (target: 0.1–1 Hz, not per-step).
        """
        latent = torch.cat([h_t, z_t.flatten(-2)], dim=-1)
        latent_tokens = self.wm_to_llm(latent)  # [batch, llm_dim]
        
        # Inject as soft prefix tokens into the LLM context
        # LLM is frozen; only the bridge projection is trained
        with torch.no_grad():
            narration = llm.generate(
                input_embeds=torch.cat([latent_tokens.unsqueeze(1), prompt_embeds], dim=1),
                max_new_tokens=128,
                temperature=0.7,
            )
        
        return narration
    
    def encode_goal(self, user_intent_text, llm):
        """
        Encode user's natural-language creative intent into WM preference space.
        Goal: translate 'write a haiku about longing in the style of Bashō'
              into a preference distribution C over WM latent states.
        """
        with torch.no_grad():
            intent_hidden = llm.encode(user_intent_text)[-1]  # last hidden state
        preference_latent = self.llm_to_wm(intent_hidden)
        return preference_latent
    
    def propose_awm(self, h_t, z_t, llm, task_description):
        """
        DECKARD-style: use LLM to propose an Abstract World Model (AWM) —
        a high-level plan as a DAG of subgoals, which the EFE planner then
        verifies/refines in imagination.
        Called at long-horizon planning checkpoints (not per-step).
        """
        latent_desc = self.narrate_latent(h_t, z_t, llm, "Describe the current creative state:")
        awm_prompt = f"""
        Current state: {latent_desc}
        Task: {task_description}
        Propose a sequence of 3-5 creative subgoals to achieve the task.
        Format as: [subgoal_1] → [subgoal_2] → ... → [final_output]
        """
        with torch.no_grad():
            awm = llm.generate(awm_prompt, max_new_tokens=256)
        return self._parse_awm(awm)
```

### 5.2 Camatkāra Narration Pipeline

```python
class CamatkaraNarrator:
    """
    At each sphurattā event, generate a narration via the LLM.
    Write a skill-library entry (Voyager-style) for future use.
    Log the event for the camatkāra evaluation protocol.
    """
    def __init__(self, vimarsa_bridge, llm, skill_library):
        self.bridge = vimarsa_bridge
        self.llm = llm
        self.library = skill_library
        self.event_log = []
    
    def on_sphuratta(self, h_t, z_t, vfe_drop, hopfield_entropy, context):
        """Called when CittaStore.is_sphuratta() fires."""
        # 1. Generate narration via LLM
        narration = self.bridge.narrate_latent(
            h_t, z_t, self.llm,
            prompt_template=VIMARSA_NARRATION_TEMPLATE
        )
        
        # 2. Compute camatkāra reward components
        delta_I_hopfield = self._compute_hopfield_info_gain(h_t, z_t)
        empowerment = self._estimate_empowerment(h_t, z_t)
        r_camatk = (
            self.alpha_1 * vfe_drop
            + self.alpha_2 * delta_I_hopfield
            + self.alpha_3 * empowerment
        )
        
        # 3. Write skill library entry
        skill_entry = {
            'description': narration,
            'latent_embedding': torch.cat([h_t, z_t.flatten(-2)], dim=-1).detach(),
            'camatk_reward': r_camatk.item(),
            'timestamp': time.time(),
            'context': context,
        }
        self.library.write(skill_entry)
        
        # 4. Log for evaluation
        self.event_log.append({
            'timestamp': time.time(),
            'narration': narration,
            'vfe_drop': vfe_drop.item(),
            'hopfield_entropy': hopfield_entropy,
            'r_camatk': r_camatk.item(),
        })
        
        return narration, r_camatk
```

---

## 6. Sleep Subsystem

### 6.1 Sleep Scheduler

```python
class SleepScheduler:
    """
    Manages NREM/REM sleep phases.
    Triggers: periodic, surprise threshold, buffer fullness, creative stagnation.
    """
    def __init__(self, world_model, actor, citta_stores, replay_buffer):
        self.wm = world_model
        self.actor = actor
        self.stores = citta_stores
        self.buffer = replay_buffer
        self.therm_budget = ThermSleepBudget(
            max_flops=1e15,  # ~1 petaFLOP per sleep phase (configurable)
            target_efficiency=0.01,  # stop when learning gain < 1% per FLOP
        )
        
    def should_sleep(self, step, vfe_history, sphuratta_history):
        """Check if sleep should be triggered."""
        periodic = (step % self.sleep_interval == 0)
        surprised = (np.mean(vfe_history[-100:]) > self.surprise_threshold)
        buffer_full = (len(self.buffer) > 0.8 * self.buffer.capacity)
        stagnant = (
            len(sphuratta_history) > 0 
            and (step - sphuratta_history[-1]) > self.stagnation_threshold
        )
        return periodic or surprised or buffer_full or stagnant
    
    def run_nrem(self):
        """NREM-analog: replay-driven consolidation."""
        print("[Sleep] Starting NREM consolidation phase...")
        self.therm_budget.reset()
        
        while not self.therm_budget.exhausted():
            # Sample prioritised batch
            batch = self.buffer.sample_prioritized(
                batch_size=32,
                alpha=0.6,  # priority exponent
            )
            
            # World model update
            loss = self.wm.world_model_loss(batch)
            loss.backward()
            self.wm_optimizer.step()
            self.wm_optimizer.zero_grad()
            
            # Update replay priorities with new TD errors
            new_priorities = self.compute_td_errors(batch)
            self.buffer.update_priorities(batch['indices'], new_priorities)
            
            # Hopfield consolidation
            for store in self.stores:
                store.consolidate_sws(self.wm)
            
            # Update thermodynamic budget
            flops_used = self.estimate_flops(loss)
            vfe_gain = self.last_vfe - loss.item()
            self.therm_budget.update(flops_used, vfe_gain)
        
        print(f"[Sleep] NREM complete. VFE: {self.wm.vfe_stats.mean:.4f}")
    
    def run_rem(self):
        """REM-analog: generative dreaming."""
        print("[Sleep] Starting REM dreaming phase...")
        self.therm_budget.reset()
        
        all_dream_latents = []
        
        while not self.therm_budget.exhausted():
            # Sample start state
            h_0 = self.sample_start_state()
            z_0_logits = self.wm.prior(h_0)
            z_0 = straight_through_sample(z_0_logits.reshape(-1, 32, 32))
            
            # Generate dream trajectory (prior only — no environmental input)
            dream_traj = self.wm.imagine_rollout(
                h_0, z_0, actor=self.actor,
                horizon=self.dream_horizon,
                mode='prior_only',
            )
            all_dream_latents.append(dream_traj['z_seq'])
            
            # Actor-critic update on dream EFE
            actor_loss = self.actor.actor_loss(self.wm, dream_traj)
            actor_loss.backward()
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()
            
            # Recognition net update (Hinton-Dayan sleep phase)
            # Train encoder to better invert the prior on dream observations
            dream_obs = dream_traj['obs_seq'].detach()
            rec_loss = self.wm.encoder_sleep_loss(dream_obs, dream_traj)
            rec_loss.backward()
            self.encoder_optimizer.step()
            
            self.therm_budget.update(
                flops=self.estimate_flops(actor_loss + rec_loss),
                gain=self.compute_dream_gain(dream_traj),
            )
        
        # Consolidate dream latents into semantic Hopfield store
        if all_dream_latents:
            dream_z = torch.cat(all_dream_latents, dim=0)
            for store in self.stores:
                store.consolidate_rem(dream_z)
        
        print(f"[Sleep] REM complete. Dream episodes: {len(all_dream_latents)}")
```

---

## 7. Camatkāra Intrinsic Reward

```python
class CamatkaraReward(nn.Module):
    """
    Intrinsic reward signal: R_camatk = α₁·ΔF + α₂·ΔI_Hopfield + α₃·Empowerment
    This is the svātantrya self-certified creative reward.
    No external judge needed.
    """
    def __init__(self, alpha_1=0.4, alpha_2=0.3, alpha_3=0.3):
        self.alpha_1 = alpha_1  # Free energy reduction weight
        self.alpha_2 = alpha_2  # Hopfield information gain weight
        self.alpha_3 = alpha_3  # Empowerment weight
        
        # Running statistics for normalisation
        self.vfe_stats = RunningStats()
        self.hopfield_stats = RunningStats()
        self.empowerment_stats = RunningStats()
        
    def compute(self, prev_vfe, curr_vfe, hopfield_store, h_t, z_t, world_model):
        """
        Compute camatkāra reward at time t.
        All components are dimensionless after normalisation.
        """
        # Component 1: VFE reduction (Friston "Eureka" signal)
        delta_F = (prev_vfe - curr_vfe).clamp(min=0)  # only reward VFE drops
        delta_F_norm = (delta_F - self.vfe_stats.mean) / (self.vfe_stats.std + 1e-8)
        
        # Component 2: Hopfield information gain
        # Approximate as the change in store entropy before/after write
        pre_entropy = hopfield_store.semantic_entropy()
        hopfield_store.write_episodic(h_t, z_t, reward=0, novelty=delta_F.item())
        post_entropy = hopfield_store.semantic_entropy()
        delta_I = (pre_entropy - post_entropy).abs()  # information gained
        delta_I_norm = (delta_I - self.hopfield_stats.mean) / (self.hopfield_stats.std + 1e-8)
        
        # Component 3: Empowerment (I[A_{t:t+k}; S_{t+k}])
        # Approximate via ensemble disagreement (cheap proxy)
        empowerment = world_model.ensemble_disagreement(h_t, z_t)
        empowerment_norm = (empowerment - self.empowerment_stats.mean) / (self.empowerment_stats.std + 1e-8)
        
        # Update running stats
        self.vfe_stats.update(delta_F.item())
        self.hopfield_stats.update(delta_I.item())
        self.empowerment_stats.update(empowerment.item())
        
        # Weighted sum
        r_camatk = (
            self.alpha_1 * delta_F_norm
            + self.alpha_2 * delta_I_norm
            + self.alpha_3 * empowerment_norm
        )
        
        return r_camatk
    
    def svātantrya_score(self, z_output, training_corpus_embeddings):
        """
        S_svātantrya: compositional novelty score.
        Distance from nearest training example in latent space.
        """
        z = z_output.flatten(-2)  # [batch, latent_dim]
        # Approximate nearest neighbour (FAISS for large corpus)
        dists = torch.cdist(z.unsqueeze(0), training_corpus_embeddings.unsqueeze(0))
        min_dist = dists.min(-1).values.squeeze(0)
        return min_dist
```

---

## 8. Pañcakṛtya Control Loop

```python
class PancakrtyaLoop:
    """
    The five-act outer control cycle of the system.
    Implemented as the main experience collection + update loop.
    """
    def __init__(self, world_model, actor, citta_stores, vimarsa_bridge,
                 sleep_scheduler, camatk_reward, skill_library):
        self.wm = world_model
        self.actor = actor
        self.stores = citta_stores
        self.vimarsa = vimarsa_bridge
        self.sleep = sleep_scheduler
        self.reward = camatk_reward
        self.library = skill_library
        self.step = 0
        
    def run(self, creative_task, n_steps=10000):
        """
        Main loop: Sṛṣṭi → Sthiti → Saṃhāra → Tirodhāna → Anugraha
        """
        h_t = self.wm.init_hidden()
        z_t = self.wm.init_latent()
        preference_C = self.vimarsa.encode_goal(creative_task.description, self.llm)
        
        for t in range(n_steps):
            # ── SṚṢṬI (creation): Imagine trajectories ──────────────────
            # At long-horizon checkpoints, get LLM AWM proposal
            if t % self.awm_interval == 0:
                awm = self.vimarsa.propose_awm(h_t, z_t, self.llm, creative_task)
                preference_C = self.update_preference_from_awm(awm, preference_C)
            
            # Select action by minimising EFE
            a_t = self.actor.act(h_t, z_t, preference_C, self.wm)
            
            # ── STHITI (maintenance): Execute and collect ────────────────
            obs_t, reward_ext, done = creative_task.step(a_t.cpu().numpy())
            h_next, z_next, lp, lq = self.wm.observe(obs_t, h_t, z_t, a_t)
            
            vfe_t = self.wm.compute_vfe(lp, lq)
            r_camatk = self.reward.compute(self.prev_vfe, vfe_t, self.stores[0], h_next, z_next, self.wm)
            total_reward = r_camatk + self.lambda_ext * reward_ext
            
            # Write to replay buffer
            self.replay_buffer.add({
                'h': h_t, 'z': z_t, 'a': a_t,
                'obs': obs_t, 'r': total_reward, 'done': done,
                'vfe': vfe_t, 'priority': vfe_t.item(),
            })
            
            # Write to Hopfield episodic store
            self.stores[0].write_episodic(h_next, z_next, total_reward, vfe_t)
            
            # ── SAṂHĀRA (compression): Compress into memory ─────────────
            if len(self.replay_buffer) > self.min_buffer_size:
                batch = self.replay_buffer.sample(self.batch_size)
                wm_loss = self.wm.world_model_loss(batch)
                wm_loss.backward()
                self.wm_optimizer.step()
                self.wm_optimizer.zero_grad()
            
            # ── TIRODHĀNA (concealment): Apply dropout/masking ───────────
            if t % self.mask_interval == 0:
                # Randomly mask portions of self-model to enable fresh inference
                # Anti-āṇava regularisation: prevent overconfident self-model
                self._apply_self_model_dropout(h_next, z_next)
            
            # ── ANUGRAHA (grace/revelation): Sphurattā detection ────────
            if self.stores[0].is_sphuratta(h_next, z_next, vfe_t):
                narration, r_event = self.vimarsa.narrator.on_sphuratta(
                    h_next, z_next, vfe_t, 
                    self.stores[0].last_retrieval_entropy,
                    context=creative_task.context,
                )
                print(f"\n[Sphurattā at step {t}]\n{narration}\n")
                self.camatk_log.append({'step': t, 'narration': narration, 'r': r_event})
            
            # Sleep trigger
            if self.sleep.should_sleep(t, self.vfe_history, self.sphuratta_times):
                self.sleep.run_nrem()
                self.sleep.run_rem()
            
            h_t, z_t = h_next, z_next
            self.prev_vfe = vfe_t
            self.step = t
```

---

## 9. Mala Regularisers

```python
class MalaRegularisers(nn.Module):
    """
    The three mala regularisers — corrective constraints on pathological attractor states.
    Applied as auxiliary losses during world model training.
    """
    def __init__(self, lambda_anava=0.01, lambda_mayiya=0.01, lambda_karma=0.01):
        self.lambda_anava = lambda_anava
        self.lambda_mayiya = lambda_mayiya
        self.lambda_karma = lambda_karma
        
    def anti_anava(self, f_self_logits):
        """
        Anti-āṇava-mala: prevent overconfident self-prior.
        Penalise low entropy in the self-model f_self(h_t, z_t).
        Ensures the system maintains epistemic humility about its own state.
        """
        # Maximise entropy of self-model predictions
        probs = F.softmax(f_self_logits, dim=-1)
        H = -(probs * probs.clamp(min=1e-8).log()).sum(-1)
        return -self.lambda_anava * H.mean()  # negative = maximise entropy
    
    def anti_mayiya(self, self_latent, world_latent):
        """
        Anti-māyīya-mala: prevent false agent/world split.
        Contrastive loss tying portions of the agent's self-encoding
        to a shared subspace of the world latent.
        Ensures the agent does not treat itself as fundamentally separate from the world.
        """
        # Cosine similarity between agent self-encoding and world latent
        # Should be moderate (not zero = isolated, not one = collapsed)
        cos_sim = F.cosine_similarity(self_latent, world_latent, dim=-1)
        # Penalise both too-high (loss of individuation) and too-low (excessive separation)
        target_sim = 0.5  # sweet spot
        return self.lambda_mayiya * (cos_sim - target_sim).pow(2).mean()
    
    def anti_karma(self, world_model, agent_action_history, batch):
        """
        Anti-kārma-mala: prevent reification of action as separate from cognition.
        Train the world model to predict past agent actions as ordinary world events.
        This ensures the agent's actions are part of the world model's ontology,
        not a separate "will" imposed from outside.
        """
        # Treat agent's past actions as observations of the world
        action_obs = batch['actions'].detach()
        action_pred = world_model.action_prediction_head(batch['h'], batch['z'])
        return self.lambda_karma * F.mse_loss(action_pred, action_obs)
```

---

## 10. Hardware Allocation on DGX Spark (128GB)

The NVIDIA DGX Spark has 128GB unified GPU memory (GB10 Blackwell chip). Budget:

| Component | GPU Memory | Notes |
|---|---|---|
| Frozen V-JEPA 2 encoder (1.2B, FP16) | ~5 GB | Feature inference only; no gradients |
| Trika WM (3 levels × ~50M params each) | ~25 GB | Main trainable parameters |
| **Nemotron 120B A12B MoE (FP4, TRT-LLM)** | **~44 GB** | **Primary āgama; 12B active params; replaces Llama-3-70B** |
| Nemotron 49B Dense (FP8, vLLM) | ~28 GB | Fast sub-agents; page-evicted during sleep |
| Vimarśa bridge (LoRA + projections) | ~0.5 GB | Trainable; small |
| EFE actor + distributional critic | ~2 GB | Trainable |
| Hopfield Citta-stores (all levels) | ~4 GB | Pattern buffers + learned prototypes |
| DIAMOND EDM decoder (Phase 5+) | ~8 GB | Optional; adds high-fidelity decoding |
| Replay buffer (priority queue) | ~6 GB | GPU-resident for fast sampling |
| Optimizer states + gradients | ~4 GB | Adam for WM-only (LLMs frozen) |
| CUDA overhead + activation cache | ~5 GB | |
| **Total (both LLMs loaded)** | **~131.5 GB** | *Exceeds 128GB — see scheduling note* |
| **Total (49B page-evicted)** | **~103.5 GB** | **24.5 GB headroom; default operating mode** |

**Memory scheduling policy**: The 49B model is active during cascade steps (cit→kriyā). During sleep cycles and vimarśa narration, the 49B's weight pages are soft-evicted using GB10 Blackwell's unified LPDDR5X zero-copy move. Both models are never fully resident simultaneously; the 49B is loaded on demand. See `PWM_Local_Models_Inference.md` for full scheduling spec.

**Training configuration:**
- `torch.compile()` for the WM forward pass (+20–40% throughput).
- `bfloat16` for WM; `FP16` for V-JEPA 2 encoder inference; FP4 for Nemotron 120B.
- Gradient checkpointing on the S4 Para level (reduces activation memory by ~60%).
- LLMs queried asynchronously (separate CUDA stream via LiteLLM) — do not block WM training.
- Sleep phases run overnight (scheduler triggers at end of the day's waking steps).

**Recommended Phase-1 minimal config (proof of concept):**
- Aparā level only (stride 1, GRU, 50M params).
- No V-JEPA 2 encoder; use BPE text tokens directly.
- No DIAMOND decoder; use linear text decoder.
- Only Nemotron 49B (no 120B MoE) for vimarśa narration in early phases.
- Total: ~65 GB — fits comfortably on the DGX Spark.

---

## 11. Key Dependencies and Repositories

```
# Core world model
pip install torch torchvision torchaudio  # PyTorch 2.x
git clone https://github.com/NM512/dreamerv3-torch  # base DreamerV3 PyTorch

# R2I S4 backbone
git clone https://github.com/state-spaces/s4  # S4 state-space model

# Hopfield networks
pip install hflayers  # ml-jku/hopfield-layers

# Active inference — EFE module only (no full POMDP rewrite)
pip install inferactively-pymdp        # pymdp.maths: compute_info_gain, compute_expected_utility
# NOTE: only pymdp.maths is used; pymdp.Agent and pymdp.envs are NOT used.
# The RSSM is the WM backbone; pymdp provides the EFE math on top of RSSM beliefs.

# LLM āgama layer — LiteLLM unified backend (local + commercial)
pip install litellm smolagents         # provider-agnostic LLM calls + agents
# LOCAL path (DGX Spark):
pip install tensorrt-llm vllm          # TRT-LLM for Nemotron 120B, vLLM for 49B
huggingface-cli download nvidia/Nemotron-3-Super-120B-A12B
huggingface-cli download meta-llama/Llama-3.3-Nemotron-Super-49B-v1.5
# COMMERCIAL path (API): set env vars ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY
# Commercial path uses LiteLLM direct — no TRT-LLM/vLLM required.
# See configs/llm_backend.yaml for full provider switching config.
# See PWM_Local_Models_Inference.md for TRT-LLM engine build.

# Context infrastructure (Pratyākṣa harness)
git clone https://github.com/SharathSPhD/pratyaksha-context-eng-harness.git \
  ~/.claude/plugins/pratyaksha-context-eng-harness
git clone https://github.com/SharathSPhD/context-engineering-harness.git
# pramāṇa fine-tuning pipeline (Phase 4+)
git clone https://github.com/SharathSPhD/pramana.git

# V-JEPA 2 encoder — DEFERRED to v2.0 (v1.0 is text-only)
# git clone https://github.com/facebookresearch/vjepa2   # uncomment for v2.0

# DIAMOND decoder (Phase 5+)
git clone https://github.com/eloialonso/diamond

# Utilities
pip install wandb hydra-core omegaconf faiss-cpu scikit-learn
pip install sentence-transformers  # for skill library embedding

# Experiment tracking
pip install mlflow  # alternative to wandb
```

---

## 12. pymdp EFE Integration with RSSM

Active inference in PWM uses `pymdp.maths` utilities as a **surgical drop-in** for the DreamerV3 actor's reward signal. The RSSM architecture is unchanged; only the policy selection criterion changes from REINFORCE to EFE.

### 12.1 Why EFE-module-only

pymdp implements a full discrete POMDP (A, B, C, D matrices with Bayesian belief propagation). The Trika RSSM uses a continuous neural generative model — replacing the RSSM with pymdp matrices would lose the WM's scalability and continuous latent structure. Instead, pymdp's math utilities are called on the RSSM's belief distribution `z_t`:

```python
# pwm/wm/efe_actor.py

import numpy as np
from pymdp.maths import compute_info_gain, compute_expected_utility

class EFEActor(nn.Module):
    """
    EFE-based policy network trained on the Expected Free Energy objective.
    
    G = ambiguity(h_t) + risk(z_t, C) − epistemic(A, z_t) − novelty(h_t, z_t)
    
    pymdp.maths functions operate on numpy arrays representing discrete beliefs.
    We treat the RSSM's 32×32 categorical posterior z_t as the belief state qs.
    """

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.ELU(),
            nn.Linear(256, action_dim),
        )
        # Preference distribution C (updated by vimarśa goal spec)
        self.register_buffer("C", torch.ones(action_dim) / action_dim)

    def compute_efe(self, h_t: torch.Tensor, z_t: torch.Tensor,
                    A: np.ndarray) -> tuple:
        """
        Compute EFE components using pymdp math on RSSM categorical beliefs.

        Parameters
        ----------
        h_t   : WM deterministic state (hidden_dim,)
        z_t   : WM stochastic state logits (stoch_dim × stoch_classes,)
        A     : likelihood matrix (n_obs × n_states) — linearised RSSM decoder
        """
        # RSSM posterior → discrete belief vector for pymdp
        qs = z_t.softmax(dim=-1).detach().cpu().numpy().flatten()   # (stoch_dim*stoch_classes,)
        C_np = self.C.cpu().numpy()

        # ① Ambiguity: E[-log p(o|s)] — decoder entropy
        ambiguity = self._decoder_entropy(h_t)

        # ② Risk: -E_Q[log p(preferred | s)] — divergence from preference C
        risk = float(-compute_expected_utility(C_np, qs))

        # ③ Epistemic value: I[s; o | π] — information gain (reduces uncertainty)
        epistemic = float(compute_info_gain(A, qs))

        # ④ Parameter novelty: KL[Q(θ) ‖ P(θ)] — approximated via running stats
        novelty = float(self._parameter_novelty(h_t, z_t))

        G = ambiguity + risk - epistemic - novelty
        return G, {"ambiguity": ambiguity, "risk": risk,
                   "epistemic": epistemic, "novelty": novelty}

    def _decoder_entropy(self, h_t: torch.Tensor) -> float:
        """Approximate ambiguity as variance of decoder output distribution."""
        with torch.no_grad():
            # Multiple forward passes under dropout → Monte Carlo entropy estimate
            preds = torch.stack([self.decoder(h_t) for _ in range(10)])
            return float(preds.var(dim=0).mean())

    def _parameter_novelty(self, h_t, z_t) -> torch.Tensor:
        """
        KL between current and reference Dirichlet concentration.
        Approximated as mean absolute deviation from running parameter mean.
        """
        state = torch.cat([h_t, z_t.flatten()], dim=-1)
        return (state - self._param_mean).abs().mean()

    def update_preference(self, C_new: torch.Tensor):
        """Called by vimarśa when user updates the goal specification."""
        self.C = C_new.to(self.C.device)

    def sample(self, h_t: torch.Tensor) -> torch.Tensor:
        """Sample action from policy network (trained to minimise EFE)."""
        logits = self.policy_net(h_t)
        return torch.distributions.Categorical(logits=logits).sample()
```

### 12.2 A-matrix Approximation

The information gain term requires a likelihood matrix A (n_obs × n_states). The RSSM decoder provides a differentiable p(o|s). The A-matrix is approximated via a linearised Jacobian evaluated at the current state:

```python
def get_likelihood_matrix(self, h_t: torch.Tensor, n_obs: int = 64) -> np.ndarray:
    """Linearised RSSM decoder Jacobian as pymdp A-matrix approximation."""
    z_basis = torch.eye(self.stoch_dim * self.stoch_classes, device=h_t.device)
    with torch.no_grad():
        obs_basis = torch.stack([
            self.decoder(torch.cat([h_t, z], dim=-1))
            for z in z_basis[:n_obs]
        ])
    A = obs_basis.softmax(dim=-1).cpu().numpy()  # (n_obs, stoch_dim*stoch_classes)
    return A
```

---

## 13. LiteLLM Provider Configuration

All LLM calls in PWM go through the `LLMBackend` class which wraps LiteLLM. A single config key switches providers. Commercial API providers bypass TRT-LLM/vLLM entirely — only LiteLLM is needed.

```yaml
# configs/llm_backend.yaml

llm:
  # Change this one key to switch the entire system's LLM backend
  provider: "nemotron-local"   # options: nemotron-local | claude-api | openai-api | gemini-api | custom

  # ── LOCAL: Nemotron on DGX Spark (default) ──────────────────────────────
  nemotron-local:
    requires: [tensorrt-llm, vllm]   # local inference engines
    primary:                          # vimarśa, memory, sleep (deep reasoning)
      model: "openai/nemotron-120b"   # LiteLLM openai-compat prefix
      api_base: "http://localhost:8000/v1"
      api_key: "local"
      temperature: 0.7
      max_tokens: 2048
    fast:                             # jñāna, kriyā (fast knowledge calls)
      model: "openai/nemotron-49b"
      api_base: "http://localhost:8001/v1"
      api_key: "local"
      temperature: 0.9
      max_tokens: 512

  # ── COMMERCIAL: Anthropic Claude ────────────────────────────────────────
  claude-api:
    requires: []     # no local inference stack
    primary:
      model: "claude-opus-4-6"
      api_key: "${ANTHROPIC_API_KEY}"
      temperature: 0.7
      max_tokens: 2048
    fast:
      model: "claude-haiku-4-5-20251001"
      api_key: "${ANTHROPIC_API_KEY}"
      temperature: 0.9
      max_tokens: 512

  # ── COMMERCIAL: OpenAI ──────────────────────────────────────────────────
  openai-api:
    requires: []
    primary:
      model: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.7
      max_tokens: 2048
    fast:
      model: "gpt-4o-mini"
      api_key: "${OPENAI_API_KEY}"
      temperature: 0.9
      max_tokens: 512

  # ── COMMERCIAL: Google Gemini ────────────────────────────────────────────
  gemini-api:
    requires: []
    primary:
      model: "gemini/gemini-2.0-flash-thinking-exp"
      api_key: "${GOOGLE_API_KEY}"
      temperature: 0.7
      max_tokens: 2048
    fast:
      model: "gemini/gemini-2.0-flash"
      api_key: "${GOOGLE_API_KEY}"
      temperature: 0.9
      max_tokens: 512

  # ── CUSTOM: Any LiteLLM-supported model ─────────────────────────────────
  custom:
    primary:
      model: "${LLM_PRIMARY_MODEL}"       # e.g. "ollama/mistral" or "together/llama-3-70b"
      api_base: "${LLM_PRIMARY_API_BASE}"
      api_key: "${LLM_PRIMARY_API_KEY}"
      temperature: 0.7
      max_tokens: 2048
    fast:
      model: "${LLM_FAST_MODEL}"
      api_base: "${LLM_FAST_API_BASE}"
      api_key: "${LLM_FAST_API_KEY}"
      temperature: 0.9
      max_tokens: 512
```

**CLI usage:**
```bash
# Run with local Nemotron (default)
python -m pwm.main "compose something surprising" --config configs/default.yaml

# Run with Claude API
python -m pwm.main "compose something surprising" --config configs/default.yaml \
  --set llm.provider=claude-api

# Run with OpenAI via CLI override
LLM_PROVIDER=openai-api python -m pwm.main "..."
```

---

*Document status: v1.1. Updated May 2026. Companion: PWM_Master_Research.md, PWM_PRD_Plan.md, PWM_MultiAgent_Architecture.md, PWM_Local_Models_Inference.md*
