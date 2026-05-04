"""
SleepAgent: NREM/REM sleep orchestrator agent.

Philosophical grounding:
  Suṣupti (Māṇḍūkya Upaniṣad 5; MV 1.2.9): Deep dreamless sleep — the state
  where individual consciousness dissolves into undifferentiated awareness.
  Kṣemarāja (PHṛ commentary, sūtra 3): Suṣupti is where saṃskāras (impressions)
  are processed and integrated without the veil of ego.

  The SleepAgent orchestrates the NREM/REM cycle, deciding WHEN to sleep and
  HOW LONG, based on the ThermSleep stopping criterion and episode completion.

  Like MemoryAgent, this is NOT a smolagents agent — it is a synchronous
  orchestrator operating on the shared call stack. It wraps SleepConsolidator
  with scheduling logic.

Architecture:
  Trigger: SleepScheduler fires (episode_count % sleep_every == 0 OR
           replay_buffer full AND vfe_plateau detected)
  Actions: NREMPhase.run_cycle() × N → REMPhase.run_cycle() × M
  Stopping: ThermSleep efficiency ratio < threshold → wake up
  Phase 4+: activates SleepConsolidator
"""

from __future__ import annotations
from typing import Any
import torch


class SleepScheduler:
    """
    Determines when to trigger a sleep episode.

    Fires when: episode count reaches interval OR VFE has plateaued
    (VFE change < plateau_threshold over last plateau_window steps).
    """

    def __init__(
        self,
        sleep_every: int = 100,          # episodes between sleep
        plateau_window: int = 20,         # steps to check VFE plateau
        plateau_threshold: float = 0.001, # min VFE change to avoid sleep
        min_buffer_size: int = 512,       # require at least N transitions
    ) -> None:
        self.sleep_every = sleep_every
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.min_buffer_size = min_buffer_size

        self._episode_count = 0
        self._vfe_history: list[float] = []
        self._sleep_count = 0

    def record_episode(self, vfe: float, buffer_size: int) -> bool:
        """Record episode completion; return True if sleep should trigger."""
        self._episode_count += 1
        self._vfe_history.append(vfe)
        if len(self._vfe_history) > self.plateau_window:
            self._vfe_history.pop(0)

        if buffer_size < self.min_buffer_size:
            return False

        # Trigger on interval
        if self._episode_count % self.sleep_every == 0:
            return True

        # Trigger on VFE plateau
        if len(self._vfe_history) >= self.plateau_window:
            vfe_range = max(self._vfe_history) - min(self._vfe_history)
            if vfe_range < self.plateau_threshold:
                return True

        return False


class SleepAgent:
    """
    Orchestrates NREM + REM sleep with scheduling and metrics logging.

    Wraps SleepConsolidator with SleepScheduler.
    Designed to be called from the PWMTrainer training loop (Phase 4+).
    """

    def __init__(
        self,
        consolidator: Any,           # SleepConsolidator
        device: torch.device | None = None,
        sleep_every: int = 100,
        log_fn: Any = None,          # callable(dict) for W&B/MLflow logging
    ) -> None:
        self.consolidator = consolidator
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scheduler = SleepScheduler(sleep_every=sleep_every)
        self.log_fn = log_fn
        self._total_sleep_episodes = 0

    def maybe_sleep(
        self,
        vfe: float,
        buffer_size: int,
        step: int,
    ) -> dict[str, Any] | None:
        """
        Check sleep trigger and run consolidation if triggered.

        Args:
            vfe: current VFE (from last training step)
            buffer_size: current replay buffer size
            step: current global training step
        Returns:
            dict of sleep metrics if sleep ran, None otherwise
        """
        if not self.scheduler.record_episode(vfe, buffer_size):
            return None

        metrics = self.consolidator.sleep(self.device)
        self._total_sleep_episodes += 1
        metrics["sleep_episode"] = self._total_sleep_episodes
        metrics["triggered_at_step"] = step

        if self.log_fn is not None:
            try:
                self.log_fn({f"sleep/{k}": v for k, v in metrics.items()})
            except Exception:
                pass

        return metrics

    def force_sleep(self, step: int) -> dict[str, Any]:
        """Force a sleep episode regardless of scheduler (e.g. at phase boundary)."""
        metrics = self.consolidator.sleep(self.device)
        self._total_sleep_episodes += 1
        metrics["sleep_episode"] = self._total_sleep_episodes
        metrics["triggered_at_step"] = step
        return metrics
