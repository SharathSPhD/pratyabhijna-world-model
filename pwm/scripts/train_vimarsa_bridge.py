"""
Sprint 13: Train VimarsaBridgeV2 on (h_t, next_token) pairs from corpus.

Loads the trained WM (step 1M, phase-2/final.pt), runs observe_step on a
fixed corpus to extract hidden states h_t, then supervised-trains the
bridge's Linear(hidden_dim, vocab_size) projection via next-token CE.

Sanskrit: vimarśa (ĪPK 1.5.11) — the reflexive turn whereby WM state shapes
LLM-space generation through learnable logit biases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pwm.perception.text import TextEncoder
from pwm.vimarsa.bridge_v2 import VimarsaBridgeV2
from pwm.world_model.trika import TrikaWorldModel

WM_CKPT = Path("/home/sharaths/projects/pwm-phase2/checkpoints/final.pt")
BRIDGE_CKPT = ROOT / "checkpoints" / "vimarsa_bridge_v2.pt"

HIDDEN_DIM = 512
OBS_DIM = 512
ACTION_DIM = 64
VOCAB_SIZE = 128256
N_STEPS = 2000
BATCH_SIZE = 32
LR = 3e-4
GRAD_CLIP = 1.0
LOG_EVERY = 100

CORPUS_TEXTS = [
    "the moon rises over the quiet lake reflecting silver light",
    "rain falls on ancient stones washing away the dust of time",
    "a song remembered from childhood floats through the evening air",
    "the river knows its way to the sea without being told",
    "stars appear one by one as darkness gathers in the east",
    "the musician's hands remember what the mind has forgotten",
    "morning mist dissolves revealing mountains that were always there",
    "the poet searches for words that already know themselves",
]


def text_to_token_ids(text: str, vocab_size: int = VOCAB_SIZE) -> torch.Tensor:
    words = text.split()
    ids = [hash(w) % vocab_size for w in words]
    return torch.tensor(ids, dtype=torch.long)


def build_pairs(wm: TrikaWorldModel, encoder: TextEncoder, device: torch.device):
    """Run WM on each corpus text, return (H, T) stacked across texts."""
    h_list: list[torch.Tensor] = []
    t_list: list[torch.Tensor] = []

    with torch.no_grad():
        for text in CORPUS_TEXTS:
            tokens = text_to_token_ids(text).to(device)
            n_words = tokens.shape[0]
            if n_words < 2:
                continue
            words = text.split()
            obs = encoder.encode_text(words, device)  # (n_words, obs_dim)
            states = wm.init_state(1, device)
            action = torch.zeros(1, ACTION_DIM, device=device)
            for step in range(n_words - 1):
                states, _, _ = wm.observe_step(
                    obs[step].unsqueeze(0), action, states, step
                )
                h_t = states[0][0].squeeze(0)  # (hidden_dim,)
                h_list.append(h_t)
                t_list.append(tokens[step + 1])

    H = torch.stack(h_list, dim=0)  # (N, hidden_dim)
    T = torch.stack(t_list, dim=0)  # (N,)
    return H, T


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[S13] device={device}")

    wm = TrikaWorldModel(
        obs_dim=OBS_DIM, action_dim=ACTION_DIM, n_levels=3,
        hidden_dim=HIDDEN_DIM, stoch_dim=32, stoch_classes=32,
        decoder_z_only=True,
    ).to(device)
    state = torch.load(WM_CKPT, map_location=device, weights_only=False)
    wm.load_state_dict(state["world_model"])
    wm.eval()
    print(f"[S13] loaded WM checkpoint step={state.get('step')}")

    encoder = TextEncoder(obs_dim=OBS_DIM).to(device)
    # Prime encoder projection by one dummy call
    _ = encoder.encode_text(["prime"], device)

    H, T = build_pairs(wm, encoder, device)
    n_pairs = H.shape[0]
    print(f"[S13] built {n_pairs} (h_t, next_token) pairs")

    bridge = VimarsaBridgeV2(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE).to(device)
    opt = torch.optim.Adam(bridge.parameters(), lr=LR)

    bridge.train()
    losses: list[float] = []
    for step in range(1, N_STEPS + 1):
        idx = torch.randint(0, n_pairs, (BATCH_SIZE,), device=device)
        h_batch = H[idx]
        t_batch = T[idx]
        logits = bridge.proj(h_batch)
        loss = F.cross_entropy(logits, t_batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bridge.parameters(), GRAD_CLIP)
        opt.step()
        losses.append(float(loss.detach()))
        if step % LOG_EVERY == 0 or step == 1:
            print(f"[S13] step {step:>5d}/{N_STEPS} loss={losses[-1]:.4f}")

    BRIDGE_CKPT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bridge.state_dict(), BRIDGE_CKPT)
    print(f"[S13] saved bridge checkpoint: {BRIDGE_CKPT}")
    print(f"[S13] initial_loss={losses[0]:.4f} final_loss={losses[-1]:.4f}")
    print("[S13] Training complete")


if __name__ == "__main__":
    main()
