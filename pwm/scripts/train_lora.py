"""
train_lora.py — Sprint 6: LoRA r=8 domain adapter training.

Philosophical grounding (CLAUDE.md §9):
  Āṇava-mala (ĪPK 3.1, Abhinavagupta): the contraction of consciousness into
  individuation. LoRA training is the antidote — it expands domain-specific
  expressiveness while the frozen W preserves the WM's universal geometry.

Training objective (Fisher LDA loss):
  Given K domains each with N_k examples, train LoRA adapters to:
    1. Maximise between-class scatter (domains far apart in adapted space)
    2. Minimise within-class scatter (same domain → compact cluster)

  L_fisher = -tr(S_W^{-1} S_B)   (maximise Fisher criterion)

  Equivalent continuous loss:
    L = Σ_{d1≠d2} exp(-||μ_{d1} - μ_{d2}||^2 / τ)   [push apart]
      + Σ_d Σ_i ||x_{d,i} - μ_d||^2                  [pull together]

  This avoids WM forward pass through the degenerate fixed-point attractor
  (Sprint 1 finding: WM energy ≈ 18 regardless of obs) and directly trains
  the adapters to maximise domain separability in the 512-dim obs space.

Corpus:
  Uses domain-representative seed phrases (no LLM calls needed).
  Each domain has 8 seed phrases covering its key vocabulary/structure.

Output:
  checkpoints/lora_step_{N}.pt   — saved every 50 steps
  checkpoints/lora_final.pt      — final checkpoint (used by engine.py)

TRIZ Principle 3 (Local Quality):
  Each domain trains its own (A, B) independently — no cross-domain
  gradient interference. Only the bank's forward() routing is shared.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[2]))
from pwm.generation.lora_adapters import DomainLoRABank  # type: ignore

# ─── Domain seed corpus ─────────────────────────────────────────────────────
# 8 short phrases per domain (no LLM calls; vocabulary-representative)
DOMAIN_SEEDS: dict[str, list[str]] = {
    "sanskrit_classical": [
        "pratyabhijñā vimarśa cit ānanda spanda sphurattā",
        "śloka anuṣṭubh chandas pāda dvitīya",
        "rasa camatkāra dhvani alaṃkāra rasāyana",
        "devanāgarī saṃskṛta mānasollāsa kāvya nāṭya",
        "ābhāsa vimarśa prakāśa citi śakti",
        "brahman ātman mokṣa karman svātantrya",
        "vedānta upaniṣad mantra japa dhyāna",
        "kavitā padya śloka gīta stotra",
    ],
    "carnatic": [
        "pallavi anupallavi caraṇam rāga tāḷa",
        "bhairavi kharaharapriya kāmbhoji shankarābharaṇam",
        "svara gamaka laya sangati kṛti",
        "ṣaḍja ṛṣabha gāndhāra madhyama pañcama",
        "ādi tāḷa rūpaka miśra cāpu",
        "thyagaraja muttusvāmi dīkṣitar syāma śāstrī",
        "shruti nāda vibration resonance divine music",
        "Śiva namaskaraṃ vandanaṃ stuti mañgaḷam",
    ],
    "hindustani": [
        "alap jod jhala vilambit drut",
        "yaman bhairavī bhairav mālkauns darbāri",
        "bandish thumrī khayal ṭhumak ṭappā",
        "tabla pakhāwaj sarod sitār bānsurī",
        "rāga mīnd kān meend gamak",
        "tīntāl ektāl jhaptāl sūltāl rūpak",
        "ustād rāg bhairav prabhātiyā sandhyā",
        "komal tīvra ṣaḍja ṛṣabha nishād",
    ],
    "western_pop": [
        "verse chorus bridge hook refrain melody",
        "electric guitar bass drum synthesizer",
        "love heartbreak dance rhythm beat",
        "chart hit single album tour encore",
        "microphone studio recording mix producer",
        "pop rock indie alternative mainstream",
        "emotion nostalgia longing joy sadness",
        "radio stream playlist concert stadium",
    ],
    "western_jazz": [
        "blue note chord resolution drone overtone swing",
        "head solo coda riff comping voicing",
        "bebop cool hard bop modal free",
        "Miles Davis Coltrane Parker Monk Mingus",
        "ii-V-I turnaround changes rhythm changes",
        "improvise phrase lick vocabulary harmonic",
        "trumpet saxophone piano bass drums quartet",
        "standard ballad blues waltz medium swing",
    ],
    "kannada_film": [
        "mukhara charana pallavi rāga ಮಳೆ ಮಣ್ಣು",
        "Rajkumar Puttanna Hunsur Krishnamurthy",
        "ಹೊಳೆ ಹೂ ಕಣ್ಣು ಹೃದಯ ಪ್ರೀತಿ",
        "nāḍu jīva prāṇa ninna namma",
        "Mysore Bengaluru Karavali Karnataka culture",
        "rainy night moon stars love longing",
        "melody tune song cinema dance",
        "Purandaradasa kīrtana vacana bhakti",
    ],
    "hindi_film": [
        "mukhra antara sanchari taal dhamaar",
        "बरसात रात दिल आँखें ज़िंदगी",
        "Lata Rafi Kishore Asha Gulzar Shailendra",
        "Mumbai Bollywood cinema romance drama",
        "ghazal thumrī bhajan classical semi-classical",
        "sitar tabla harmonium bansuri sarangi",
        "pyaar dil mohabbat ishq bewafā",
        "filmi gana superhit evergreen classic",
    ],
    "tamil_classical": [
        "tinai akam puram kuyil kadal",
        "sangam aham puram Tolkāppiyam",
        "Tirukkuṟaḷ couplet ethical virtue wisdom",
        "Murugan Karthikeya Kuṟinci hill jasmine",
        "classical Carnatic south Indian rāga",
        "padam jāvaḷi pada kīrtanai",
        "bhakti devotion Āḷvār Nāyaṉmār",
        "kuṟuntokai natṟiṇai Akananūṟu",
    ],
    "telugu_padyam": [
        "padyamu nadi sandhya pakshulu dīpaṃ",
        "Annamayya Tyagaraja Kshetrayya padamu",
        "Telugu literature kāvya Mahābhārata Rāmāyaṇa",
        "Andhra Telangana river Krishna Godavari",
        "నది సంధ్య పక్షులు దీపం చెట్టు",
        "classical metre matta ebha champaka",
        "devotion bhakti natarāja krishna radha",
        "prabandha āhvāna maṅgaḷam stuti",
    ],
    "bengali_lyric": [
        "basanta phul batas alo rabindra",
        "Tagore Nazrul baul kirtan bhatiyali",
        "বসন্ত ফুল বাতাস আলো নদী",
        "monsoon padma river Bengal delta",
        "romantic lyric nature village rural",
        "sitar esraj dotara baul instrument",
        "Durga Puja festival season autumn",
        "bauliana maijbhandari spirituality",
    ],
    "english_romantic": [
        "autumn dew mist lake twilight silence",
        "Keats Shelley Wordsworth Byron Coleridge",
        "ode sonnet elegy lyric ballad",
        "nature sublime transcendence beauty truth",
        "nightingale grecian urn autumn ode",
        "shadow gold ripple haze reflection",
        "eternal momentary permanence fleeting",
        "love loss memory dream reverie",
    ],
    "english_modernist": [
        "fragment interior window corridor light concrete",
        "Eliot Pound Woolf Joyce Stevens",
        "stream of consciousness montage collage",
        "urban alienation industrial wasteland",
        "April cruelest lilacs breeding memory",
        "glass pause drift absence threshold",
        "objectivism imagism vorticism Dada",
        "modern city crowd anonymous voice",
    ],
    "english_beat": [
        "neon exhaust street diner dawn laughter",
        "Ginsberg Kerouac Ferlinghetti Corso Snyder",
        "howl dharma bum road jazz dharma",
        "San Francisco New York bohemian",
        "taxi jazz drum smoke cigarette coffee",
        "highway America spontaneous prose",
        "Buddhist Zen enlightenment impermanence",
        "freedom rebellion nonconformity spirit",
    ],
    "world_fusion": [
        "sea shore tide migration threshold wave",
        "horizon salt boat wind diaspora",
        "multicultural blend hybrid tradition",
        "Middle Eastern African Latin Celtic",
        "oud kora djembe sitar tabla fado",
        "immigrant language memory homeland",
        "intercultural dialogue border crossing",
        "global south roots identity belonging",
    ],
    "generic": [
        "image sound light shadow motion texture",
        "rhythm pattern structure form beauty",
        "creative expression art form voice",
        "time space memory imagination dream",
        "emotion thought feeling perception",
        "music poetry dance drama painting",
        "narrative metaphor symbol archetype",
        "consciousness experience reality being",
    ],
}


# ─── Synthetic TextEncoder (matches pwm.perception.text.TextEncoder) ─────────

class _BagOfWordsEncoder(torch.nn.Module):
    """
    Deterministic bag-of-words encoder mapping text → 512-dim float32 vector.

    Used when pwm.perception.text.TextEncoder is unavailable (no GPU/too slow).
    Each word hashes to a sparse vector; sentences sum+L2-normalise.
    The embedding is stable (no random state) and captures vocabulary overlap.
    """

    def __init__(self, obs_dim: int = 512) -> None:
        super().__init__()
        self.obs_dim = obs_dim

    def forward(self, texts: list[str], device: torch.device | None = None) -> torch.Tensor:
        device = device or torch.device("cpu")
        batch = torch.zeros(len(texts), self.obs_dim, device=device)
        for i, text in enumerate(texts):
            for wi, word in enumerate(text.lower().split()):
                idx = (hash(word) ^ (wi * 2654435761)) % self.obs_dim
                batch[i, idx] += 1.0
        # L2-normalise each row
        norms = batch.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return batch / norms


def _load_text_encoder(obs_dim: int, device: torch.device) -> torch.nn.Module:
    """Load real TextEncoder if available, else fall back to BoW."""
    try:
        from pwm.perception.text import TextEncoder  # type: ignore
        enc = TextEncoder(obs_dim=obs_dim).to(device).eval()
        print("  [LoRA train] Using real TextEncoder (sentence-transformers)")
        return enc
    except Exception as e:
        print(f"  [LoRA train] TextEncoder unavailable ({e}); using BoW fallback")
        return _BagOfWordsEncoder(obs_dim).to(device).eval()


# ─── Fisher LDA loss ─────────────────────────────────────────────────────────

def fisher_lda_loss(
    embeddings: torch.Tensor,     # (N, D)
    labels: torch.Tensor,         # (N,) int64 domain indices
    tau: float = 0.1,
    within_weight: float = 1.0,
    between_weight: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Continuous Fisher LDA loss.

    between_loss: penalise close class means (push domains apart)
    within_loss:  penalise spread within each class (pull same-domain together)

    L = between_weight * between_loss + within_weight * within_loss
    """
    device = embeddings.device
    K = int(labels.max().item()) + 1
    D = embeddings.shape[1]

    # Per-class means
    means = torch.zeros(K, D, device=device)
    counts = torch.zeros(K, device=device)
    for k in range(K):
        mask = labels == k
        if mask.any():
            means[k] = embeddings[mask].mean(0)
            counts[k] = mask.sum().float()

    # Between-class: push class means apart
    # Use mean-field softmax repulsion over all pairs
    valid = counts > 0
    valid_means = means[valid]                      # (K', D)
    # Pairwise distances
    diff = valid_means.unsqueeze(0) - valid_means.unsqueeze(1)   # (K', K', D)
    sq_dist = (diff * diff).sum(-1)                             # (K', K')
    # Only upper triangle (no self-pairs)
    mask_upper = torch.triu(torch.ones(sq_dist.shape, device=device, dtype=torch.bool), diagonal=1)
    between_loss = torch.exp(-sq_dist[mask_upper] / tau).mean()

    # Within-class: pull same-domain embeddings together
    within_loss = torch.zeros(1, device=device)
    for k in range(K):
        mask_k = labels == k
        if mask_k.sum() > 1:
            diff_k = embeddings[mask_k] - means[k].detach()
            within_loss = within_loss + (diff_k * diff_k).sum(-1).mean()
    within_loss = within_loss / max(1, K)

    total_loss = between_weight * between_loss + within_weight * within_loss

    return total_loss, {
        "between_loss": float(between_loss.item()),
        "within_loss": float(within_loss.item()),
        "total_loss": float(total_loss.item()),
    }


# ─── Training loop ────────────────────────────────────────────────────────────

def train(
    obs_dim: int = 512,
    r: int = 8,
    alpha: float = 16.0,
    lr: float = 3e-4,
    n_steps: int = 200,
    save_every: int = 50,
    device_str: str = "cuda",
    checkpoint_in: str | None = None,
    checkpoint_out: str = "checkpoints/lora_final.pt",
    tau: float = 0.15,
) -> dict:
    """Train DomainLoRABank using Fisher LDA loss."""
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"\n[LoRA train] Device: {device} | r={r} | α={alpha} | lr={lr} | steps={n_steps}")

    # ── Build bank + encoder ──────────────────────────────────────────────────
    if checkpoint_in and Path(checkpoint_in).exists():
        bank = DomainLoRABank.load(Path(checkpoint_in), obs_dim=obs_dim).to(device)
        print(f"  Resumed from {checkpoint_in}")
    else:
        bank = DomainLoRABank(obs_dim=obs_dim, r=r, alpha=alpha).to(device)
    bank.freeze_base_weights()

    encoder = _load_text_encoder(obs_dim, device)

    trainable = bank.trainable_parameters()
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    optimiser = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=n_steps, eta_min=lr * 0.01)

    # ── Precompute embeddings (no grad) ───────────────────────────────────────
    print("  Encoding domain seed texts…")
    domain_list = list(DOMAIN_SEEDS.keys())
    all_texts: list[str] = []
    all_labels: list[int] = []

    for di, domain in enumerate(domain_list):
        seeds = DOMAIN_SEEDS[domain]
        all_texts.extend(seeds)
        all_labels.extend([di] * len(seeds))

    with torch.no_grad():
        raw_emb = encoder(all_texts, device=device)          # (N, obs_dim)

    labels_t = torch.tensor(all_labels, dtype=torch.long, device=device)  # (N,)
    N = len(all_texts)
    print(f"  Corpus: {N} phrases × {len(domain_list)} domains")

    # ── Training ──────────────────────────────────────────────────────────────
    losses: list[float] = []
    ckpt_dir = Path(checkpoint_out).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    for step in range(1, n_steps + 1):
        optimiser.zero_grad(set_to_none=True)

        # Apply per-example domain adapter
        adapted = torch.zeros_like(raw_emb)
        for di, domain in enumerate(domain_list):
            mask = labels_t == di
            if mask.any():
                adapted[mask] = bank(domain, raw_emb[mask])

        loss, log = fisher_lda_loss(adapted, labels_t, tau=tau)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimiser.step()
        scheduler.step()

        losses.append(log["total_loss"])

        if step % 20 == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            print(f"  Step {step:4d}/{n_steps} | loss={log['total_loss']:.4f} "
                  f"(between={log['between_loss']:.4f}, within={log['within_loss']:.4f}) "
                  f"| lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s")

        if step % save_every == 0:
            ckpt = ckpt_dir / f"lora_step_{step:06d}.pt"
            bank.save(ckpt)
            print(f"  ✓ Saved {ckpt}")

    # Final save
    final_ckpt = Path(checkpoint_out)
    bank.save(final_ckpt)
    elapsed = time.perf_counter() - t0

    result = {
        "n_steps": n_steps,
        "obs_dim": obs_dim,
        "r": r,
        "alpha": alpha,
        "lr": lr,
        "final_loss": losses[-1] if losses else 0.0,
        "initial_loss": losses[0] if losses else 0.0,
        "loss_reduction": (losses[0] - losses[-1]) / max(1e-8, losses[0]) if losses else 0.0,
        "elapsed_s": round(elapsed, 1),
        "checkpoint": str(final_ckpt),
        "device": str(device),
        "trainable_params": sum(p.numel() for p in trainable),
    }
    print(f"\n✓ Training complete in {elapsed:.1f}s")
    print(f"  Loss: {losses[0]:.4f} → {losses[-1]:.4f} "
          f"(reduction: {result['loss_reduction']*100:.1f}%)")
    print(f"  Final checkpoint: {final_ckpt}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DomainLoRABank (Sprint 6)")
    parser.add_argument("--obs-dim", type=int, default=512)
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tau", type=float, default=0.15,
                        help="Temperature for between-class repulsion loss")
    parser.add_argument("--checkpoint-in", type=str, default=None)
    parser.add_argument("--checkpoint-out", type=str,
                        default="checkpoints/lora_final.pt")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Save training result to JSON file")
    args = parser.parse_args()

    result = train(
        obs_dim=args.obs_dim,
        r=args.r,
        alpha=args.alpha,
        lr=args.lr,
        n_steps=args.n_steps,
        save_every=args.save_every,
        device_str=args.device,
        checkpoint_in=args.checkpoint_in,
        checkpoint_out=args.checkpoint_out,
        tau=args.tau,
    )

    if args.output_json:
        Path(args.output_json).write_text(json.dumps(result, indent=2))
        print(f"  Result saved to {args.output_json}")


if __name__ == "__main__":
    main()
